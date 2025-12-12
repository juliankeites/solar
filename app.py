import json
import warnings
from datetime import datetime, date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

warnings.filterwarnings("ignore")

# --- YOUR LIVE SYSTEM DEFAULTS ---
LAT = 51.32
LON = -0.56
LIVE_PEAK_KW = 1.8

# Battery system defaults (Tesla Powerwall 3)
BATTERY_CAPACITY_KWH = 13.5  # Tesla Powerwall 3
BATTERY_EFFICIENCY = 0.95
BATTERY_MAX_CHARGE_RATE_KW = 5.0
BATTERY_MAX_DISCHARGE_RATE_KW = 5.0
BASELOAD_KW = 0.4
MIN_SOLAR_THRESHOLD = 50  # W/m² minimum for production

arrays = [
    {"name": "NE", "n_modules": 7, "tilt": 45, "azimuth": 30, "loss_factor": 0.85},
    {"name": "East", "n_modules": 7, "tilt": 45, "azimuth": 115, "loss_factor": 0.85},
    {"name": "South", "n_modules": 5, "tilt": 45, "azimuth": 210, "loss_factor": 0.85},
    {"name": "West", "n_modules": 8, "tilt": 45, "azimuth": 296, "loss_factor": 0.85},
]

P_MODULE_WP = 440
TOTAL_KWP = 11.88
SYSTEM_EFFICIENCY = 0.85


class EnhancedSolarEstimator:
    """Enhanced solar production estimator with multiple methods"""

    def __init__(self, lat, lon, arrays):
        self.lat = lat
        self.lon = lon
        self.arrays = arrays
        self.total_kwp = sum(a["n_modules"] * P_MODULE_WP / 1000 for a in arrays)

    @st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
    def get_comprehensive_weather_data(_self, days=3):
        """Get comprehensive weather data with multiple parameters"""
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={_self.lat}&longitude={_self.lon}"
            f"&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,"
            f"surface_pressure,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,"
            f"direct_radiation,diffuse_radiation,global_tilted_irradiance,"
            f"direct_normal_irradiance,shortwave_radiation,precipitation,"
            f"wind_speed_10m,wind_direction_10m,is_day"
            f"&timezone=Europe%2FLondon&forecast_days={days}"
        )

        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            hourly = data.get("hourly", {})

            times = pd.to_datetime(hourly["time"])

            # Basic irradiance data
            direct_rad = np.array(hourly.get("direct_radiation", [0] * len(times)))
            diffuse_rad = np.array(hourly.get("diffuse_radiation", [0] * len(times)))
            dni = np.array(hourly.get("direct_normal_irradiance", [0] * len(times)))
            gti = np.array(hourly.get("global_tilted_irradiance", [0] * len(times)))
            
            # Day/night indicator
            is_day = np.array(hourly.get("is_day", [1] * len(times)))

            # Weather parameters
            cloud_cover = np.array(hourly.get("cloud_cover", [0] * len(times)))
            temperature = np.array(hourly.get("temperature_2m", [15] * len(times)))
            humidity = np.array(hourly.get("relative_humidity_2m", [60] * len(times)))
            wind_speed = np.array(hourly.get("wind_speed_10m", [2] * len(times)))
            pressure = np.array(hourly.get("surface_pressure", [1013] * len(times)))

            # Calculate GHI
            ghi = np.maximum(0, direct_rad + diffuse_rad)

            return {
                "times": times,
                "ghi": ghi,
                "dni": dni,
                "dhi": diffuse_rad,
                "gti": gti,
                "cloud_cover": cloud_cover,
                "temperature": temperature,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "pressure": pressure,
                "direct_rad": direct_rad,
                "diffuse_rad": diffuse_rad,
                "is_day": is_day,
            }

        except Exception as e:
            st.warning(f"Weather API failed: {str(e)}. Using simulated data.")
            # Fallback to simulated data if API fails
            return _self._generate_simulated_data(days)

    def _generate_simulated_data(self, days):
        """Generate realistic simulated data"""
        n_points = days * 24
        # Start from current time
        now = datetime.now()
        start_time = datetime(now.year, now.month, now.day, now.hour)
        times = pd.date_range(start=start_time, periods=n_points, freq="H")

        t = np.linspace(0, days * 2 * np.pi, n_points)

        # Generate day/night cycle based on time of day
        hour_of_day = times.hour.values
        is_day = (hour_of_day >= 6) & (hour_of_day <= 20)  # 6 AM to 8 PM

        day_pattern = np.sin(np.linspace(0, np.pi, 24)) ** 2
        day_pattern = np.tile(day_pattern, days)

        cloud_noise = 0.3 * np.sin(t / 3) + 0.2 * np.sin(t / 7) + 0.1 * np.random.randn(
            n_points
        )
        cloud_cover = np.clip(30 + 40 * (cloud_noise + 1) / 2, 0, 100)

        ghi_clear = 1000 * day_pattern * np.clip(1 - 0.1 * np.sin(t / 10), 0.7, 1)
        cloud_factor = 1 - 0.8 * (cloud_cover / 100) ** 1.5
        ghi = ghi_clear * cloud_factor
        # Zero out night hours
        ghi = np.where(is_day, ghi, 0)

        temperature = 15 + 10 * day_pattern + 5 * np.sin(t / 10)

        return {
            "times": times,
            "ghi": ghi,
            "dni": ghi * 0.7,
            "dhi": ghi * 0.3,
            "gti": ghi * 1.1,
            "cloud_cover": cloud_cover,
            "temperature": temperature,
            "humidity": 60 + 20 * np.sin(t / 5),
            "wind_speed": 2 + 3 * day_pattern,
            "pressure": 1013 * np.ones_like(ghi),
            "direct_rad": ghi * 0.6,
            "diffuse_rad": ghi * 0.4,
            "is_day": is_day.astype(int),
        }

    def calculate_solar_position(self, times):
        timestamps = pd.to_datetime(times)
        doy = timestamps.dayofyear.values

        B = 2 * np.pi * (doy - 1) / 365
        delta = (
            0.006918
            - 0.399912 * np.cos(B)
            + 0.070257 * np.sin(B)
            - 0.006758 * np.cos(2 * B)
            + 0.000907 * np.sin(2 * B)
            - 0.002697 * np.cos(3 * B)
            + 0.001480 * np.sin(3 * B)
        )
        delta = np.degrees(delta)

        E = 229.2 * (
            0.000075
            + 0.001868 * np.cos(B)
            - 0.032077 * np.sin(B)
            - 0.014615 * np.cos(2 * B)
            - 0.040849 * np.sin(2 * B)
        )

        lst = timestamps.hour.values + timestamps.minute.values / 60
        t_corr = 4 * (self.lon - 0) + E
        solar_time = lst + t_corr / 60

        omega = 15 * (solar_time - 12)

        lat_rad = np.radians(self.lat)
        delta_rad = np.radians(delta)
        omega_rad = np.radians(omega)

        cos_theta_z = (
            np.sin(lat_rad) * np.sin(delta_rad)
            + np.cos(lat_rad) * np.cos(delta_rad) * np.cos(omega_rad)
        )
        cos_theta_z = np.clip(cos_theta_z, 0, 1)
        theta_z = np.degrees(np.arccos(cos_theta_z))

        sin_alpha = (
            np.cos(delta_rad)
            * np.sin(omega_rad)
            / (np.sin(np.radians(theta_z)) + 1e-6)
        )
        sin_alpha = np.clip(sin_alpha, -1, 1)
        cos_alpha = (
            np.sin(delta_rad) * np.cos(lat_rad)
            - np.cos(delta_rad) * np.sin(lat_rad) * np.cos(omega_rad)
        ) / (np.sin(np.radians(theta_z)) + 1e-6)

        alpha = np.degrees(np.arctan2(sin_alpha, cos_alpha))
        alpha = np.where(alpha < 0, alpha + 360, alpha)

        G_sc = 1367
        G_on = G_sc * (1 + 0.033 * np.cos(np.radians(360 * doy / 365)))

        return {
            "zenith": theta_z,
            "azimuth": alpha,
            "cos_zenith": cos_theta_z,
            "declination": delta,
            "hour_angle": omega,
            "extraterrestrial": G_on * cos_theta_z,
        }

    def calculate_sunrise_sunset(self, times):
        """Calculate sunrise and sunset times for each day"""
        solar_pos = self.calculate_solar_position(times)
        zenith = solar_pos["zenith"]
        
        # Find hours where sun is above horizon (zenith < 90)
        sun_up = zenith < 90
        
        # Group by day
        df = pd.DataFrame({
            'time': times,
            'sun_up': sun_up,
            'zenith': zenith
        })
        df['date'] = df['time'].dt.date
        
        # Calculate sunrise/sunset for each day
        sunrise_sunset = {}
        for date_val, group in df.groupby('date'):
            day_hours = group[group['sun_up']]
            if len(day_hours) > 0:
                sunrise = day_hours['time'].min()
                sunset = day_hours['time'].max()
                sunrise_sunset[date_val] = {
                    'sunrise': sunrise,
                    'sunset': sunset,
                    'daylight_hours': len(day_hours),
                    'avg_zenith': day_hours['zenith'].mean()
                }
        
        return sunrise_sunset

    def clearsky_model_ineichen(self, solar_pos, pressure=1013):
        zenith = solar_pos["zenith"]
        cos_zenith = solar_pos["cos_zenith"]
        G_on = solar_pos["extraterrestrial"] / np.maximum(cos_zenith, 1e-6)

        air_mass = 1 / (cos_zenith + 0.50572 * (96.07995 - zenith) ** -1.6364)
        air_mass = np.clip(air_mass, 1, 10)
        air_mass *= pressure / 1013

        fh1 = np.exp(-pressure / 8000)
        fh2 = np.exp(-pressure / 1250)

        c = 0.056 * air_mass * fh1
        _b = 0.4 * fh2
        dni_clear = G_on * np.exp(-0.09 * air_mass * (fh1 + fh2))

        dhi_clear = G_on * (0.006 + 0.045 * (1 - np.exp(-air_mass * c)))

        ghi_clear = dni_clear * cos_zenith + dhi_clear

        mask = zenith < 90
        ghi_clear = np.where(mask, ghi_clear, 0)
        dni_clear = np.where(mask, dni_clear, 0)
        dhi_clear = np.where(mask, dhi_clear, 0)

        return {
            "ghi": ghi_clear,
            "dni": dni_clear,
            "dhi": dhi_clear,
            "air_mass": air_mass,
        }

    def cloud_transmission_model(self, ghi_clear, cloud_cover, solar_pos):
        zenith = solar_pos["zenith"]
        cos_zenith = np.maximum(solar_pos["cos_zenith"], 1e-3)

        cloud_optical_depth = 0.1 + 10 * (cloud_cover / 100) ** 2

        transmission = np.exp(-cloud_optical_depth / cos_zenith)

        min_transmission = 0.1 + 0.2 * (1 - cloud_cover / 100)
        transmission = np.clip(transmission, min_transmission, 1)

        ghi_cloudy = ghi_clear * transmission

        ghi_cloudy = np.where(
            cloud_cover > 50,
            ghi_cloudy * (1 + 0.1 * (cloud_cover - 50) / 50),
            ghi_cloudy,
        )

        return ghi_cloudy, transmission

    def pvwatts_model(self, ghi, temperature, wind_speed, solar_pos):
        T_ref = 25
        G_ref = 1000
        gamma = -0.004

        # Only generate power during daylight hours with sufficient irradiance
        mask = (ghi > MIN_SOLAR_THRESHOLD) & (solar_pos["zenith"] < 90)
        
        T_module = temperature + ghi * np.exp(-3.47 - 0.0594 * wind_speed) / 1000

        power_dc = (
            self.total_kwp
            * 1000
            * (ghi / G_ref)
            * (1 + gamma * (T_module - T_ref))
        )
        power_dc = np.where(mask, power_dc, 0)

        power_dc_per_unit = power_dc / (self.total_kwp * 1000 + 1e-6)
        inv_eff = 0.96 - 0.05 * power_dc_per_unit + 0.04 * power_dc_per_unit ** 2
        inv_eff = np.clip(inv_eff, 0.9, 0.965)

        power_ac = power_dc * inv_eff * SYSTEM_EFFICIENCY
        power_ac = np.clip(power_ac, 0, self.total_kwp * 1000 * 1.1)

        return power_ac, T_module, inv_eff

    def perez_model_poa(self, weather_data, solar_pos):
        ghi = weather_data["ghi"]
        dni = weather_data["dni"]
        dhi = weather_data["dhi"]

        zenith = solar_pos["zenith"]
        azimuth = solar_pos["azimuth"]

        total_poa = np.zeros_like(ghi)

        for array in self.arrays:
            tilt = array["tilt"]
            array_azimuth = array["azimuth"]

            tilt_rad = np.radians(tilt)
            zenith_rad = np.radians(zenith)

            cos_aoi = (
                np.cos(zenith_rad) * np.cos(tilt_rad)
                + np.sin(zenith_rad)
                * np.sin(tilt_rad)
                * np.cos(np.radians(azimuth - array_azimuth))
            )
            cos_aoi = np.clip(cos_aoi, 0, 1)

            f_sky = (1 + np.cos(tilt_rad)) / 2
            f_ground = (1 - np.cos(tilt_rad)) / 2

            albedo = 0.2

            poa_beam = dni * cos_aoi
            poa_diffuse = dhi * f_sky
            poa_ground = ghi * albedo * f_ground

            poa_total = poa_beam + poa_diffuse + poa_ground

            array_kwp = array["n_modules"] * P_MODULE_WP / 1000
            array_power = (
                poa_total / 1000 * array_kwp * array["loss_factor"] * 0.96
            )

            total_poa += array_power * 1000

        return total_poa

    def machine_learning_correction(self, predicted_power, historical_ratio=None):
        hour_of_day = np.arange(len(predicted_power)) % 24

        morning_mask = (hour_of_day >= 8) & (hour_of_day <= 11)
        predicted_power[morning_mask] *= 0.95

        afternoon_mask = (hour_of_day >= 13) & (hour_of_day <= 16)
        predicted_power[afternoon_mask] *= 1.05

        if historical_ratio is not None:
            predicted_power *= historical_ratio

        return predicted_power

    def estimate_production(self, weather_data, method="combined"):
        solar_pos = self.calculate_solar_position(weather_data["times"])

        p_pvwatts, T_module, inv_eff = self.pvwatts_model(
            weather_data["ghi"],
            weather_data["temperature"],
            weather_data["wind_speed"],
            solar_pos,
        )

        p_perez = self.perez_model_poa(weather_data, solar_pos)

        clearsky = self.clearsky_model_ineichen(
            solar_pos, weather_data["pressure"].mean()
        )
        ghi_cloudy, cloud_trans = self.cloud_transmission_model(
            clearsky["ghi"], weather_data["cloud_cover"], solar_pos
        )
        p_cloud_adj, _, _ = self.pvwatts_model(
            ghi_cloudy,
            weather_data["temperature"],
            weather_data["wind_speed"],
            solar_pos,
        )

        if method == "pvwatts":
            final_power = p_pvwatts
        elif method == "perez":
            final_power = p_perez
        elif method == "cloud":
            final_power = p_cloud_adj
        elif method == "combined":
            weights = {"pvwatts": 0.3, "perez": 0.4, "cloud": 0.3}
            final_power = (
                weights["pvwatts"] * p_pvwatts
                + weights["perez"] * p_perez
                + weights["cloud"] * p_cloud_adj
            )
        else:
            final_power = p_pvwatts

        final_power = self.machine_learning_correction(final_power)
        final_power = np.clip(final_power, 0, self.total_kwp * 1000)

        return {
            "power": final_power,
            "pvwatts": p_pvwatts,
            "perez": p_perez,
            "cloud_adj": p_cloud_adj,
            "temperature": T_module,
            "inverter_eff": inv_eff,
            "cloud_transmission": cloud_trans,
            "solar_zenith": solar_pos["zenith"],
        }

    def calculate_expected_max(self, days=7):
        """Clear-sky theoretical maximum over a horizon, with daily series"""
        now = datetime.now()
        start_time = datetime(now.year, now.month, now.day, 0, 0, 0)
        times = pd.date_range(start=start_time, periods=days * 24, freq="H")

        ideal_weather = {
            "times": times,
            "ghi": np.zeros(len(times)),
            "temperature": 20 * np.ones(len(times)),
            "cloud_cover": np.zeros(len(times)),
            "wind_speed": 2 * np.ones(len(times)),
            "pressure": 1013 * np.ones(len(times)),
        }

        solar_pos = self.calculate_solar_position(times)
        clearsky = self.clearsky_model_ineichen(solar_pos)

        ideal_weather["ghi"] = clearsky["ghi"]
        ideal_weather["dni"] = clearsky["dni"]
        ideal_weather["dhi"] = clearsky["dhi"]

        # Use combined method on clear-sky GHI to get "max" production
        max_estimation = self.estimate_production(ideal_weather, method="combined")

        df = pd.DataFrame(
            {"power": max_estimation["power"], "ghi": ideal_weather["ghi"]},
            index=times,
        )

        daily_max_kw = df["power"].resample("D").max() / 1000.0
        daily_energy_kwh = df["power"].resample("D").sum() / 1000.0

        return {
            "daily_max_kw": daily_max_kw,
            "daily_energy_kwh": daily_energy_kwh,
            "total_max_kw": daily_max_kw.max(),
            "avg_daily_energy": daily_energy_kwh.mean(),
        }

    def simulate_battery_operation(self, solar_power_kw, baseload_kw, 
                                   battery_capacity_kwh, initial_soc_percent=50,
                                   max_charge_rate_kw=5.0, max_discharge_rate_kw=5.0):
        """
        Simulate battery operation with self-consumption optimization
        Returns: battery_soc, grid_import, grid_export, self_consumption
        """
        n = len(solar_power_kw)
        battery_soc = np.zeros(n)
        grid_import = np.zeros(n)
        grid_export = np.zeros(n)
        self_consumption = np.zeros(n)
        
        # Convert initial SOC to kWh
        current_soc_kwh = battery_capacity_kwh * initial_soc_percent / 100
        
        for i in range(n):
            # Current solar generation
            solar_gen = solar_power_kw[i]
            
            # Calculate net load (positive = needs power, negative = excess)
            net_load = baseload_kw - solar_gen
            
            if net_load < 0:  # Excess solar
                excess = -net_load
                
                # Charge battery with excess solar
                charge_possible = min(
                    excess,
                    max_charge_rate_kw,
                    (battery_capacity_kwh - current_soc_kwh) / 1.0  # 1 hour timestep
                )
                charge_energy = charge_possible * BATTERY_EFFICIENCY
                
                current_soc_kwh += charge_energy
                self_consumption[i] = solar_gen
                
                # Remaining excess goes to grid
                grid_export[i] = excess - charge_possible
                grid_import[i] = 0
                
            else:  # Solar deficit
                deficit = net_load
                
                # Try to discharge battery
                discharge_possible = min(
                    deficit,
                    max_discharge_rate_kw,
                    current_soc_kwh / 1.0  # 1 hour timestep
                )
                
                if discharge_possible > 0:
                    current_soc_kwh -= discharge_possible
                    deficit -= discharge_possible
                    self_consumption[i] = solar_gen
                else:
                    self_consumption[i] = solar_gen
                
                # Remaining deficit comes from grid
                grid_import[i] = deficit
                grid_export[i] = 0
            
            # Ensure SOC stays within bounds
            current_soc_kwh = np.clip(current_soc_kwh, 0, battery_capacity_kwh)
            battery_soc[i] = current_soc_kwh
            
        return {
            'battery_soc_kwh': battery_soc,
            'battery_soc_percent': battery_soc / battery_capacity_kwh * 100,
            'grid_import_kw': grid_import,
            'grid_export_kw': grid_export,
            'self_consumption_kw': self_consumption,
            'solar_to_battery_kwh': np.sum(np.maximum(0, solar_power_kw - baseload_kw)),
            'total_grid_import_kwh': np.sum(grid_import),
            'total_grid_export_kwh': np.sum(grid_export),
            'self_sufficiency_percent': np.sum(self_consumption) / (np.sum(solar_power_kw) + 1e-6) * 100
        }


def run_app():
    st.set_page_config(
        page_title="Enhanced Solar Production Estimator",
        layout="wide",
    )

    st.title("🏠 Enhanced Solar Production with Battery Storage")

    with st.sidebar:
        st.header("⚙️ Configuration")
        lat = st.number_input("Latitude", value=LAT, format="%.4f")
        lon = st.number_input("Longitude", value=LON, format="%.4f")
        
        st.subheader("🔋 Battery System")
        battery_capacity = st.number_input(
            "Battery capacity (kWh)", 
            value=BATTERY_CAPACITY_KWH,
            min_value=0.0,
            max_value=100.0,
            step=0.1
        )
        initial_soc = st.slider(
            "Initial State of Charge (%)", 
            0, 100, 50
        )
        max_charge_rate = st.number_input(
            "Max charge rate (kW)", 
            value=BATTERY_MAX_CHARGE_RATE_KW,
            min_value=0.0,
            max_value=20.0,
            step=0.1
        )
        max_discharge_rate = st.number_input(
            "Max discharge rate (kW)", 
            value=BATTERY_MAX_DISCHARGE_RATE_KW,
            min_value=0.0,
            max_value=20.0,
            step=0.1
        )
        
        st.subheader("🏠 House Load")
        baseload = st.number_input(
            "Baseload (kW)", 
            value=BASELOAD_KW,
            min_value=0.0,
            max_value=10.0,
            step=0.1
        )
        
        st.subheader("📈 Forecast Settings")
        days = st.slider("Forecast days", min_value=1, max_value=7, value=3)
        show_night = st.checkbox("Show night hours", value=False)
        
        method = st.selectbox(
            "Default method to highlight",
            options=["combined", "pvwatts", "perez", "cloud"],
            index=0,
        )

    estimator = EnhancedSolarEstimator(lat, lon, arrays)

    st.subheader("📊 Weather data and predictions")
    with st.spinner("Fetching weather data and running models..."):
        weather_data = estimator.get_comprehensive_weather_data(days=days)
        max_calc = estimator.calculate_expected_max(days=days)

        methods = ["pvwatts", "perez", "cloud", "combined"]
        results = {
            m: estimator.estimate_production(weather_data, method=m) for m in methods
        }

    # Calculate sunrise/sunset times
    sunrise_sunset = estimator.calculate_sunrise_sunset(weather_data["times"])
    
    # Use as much of the horizon as available
    times = weather_data["times"]
    df_comparison = pd.DataFrame(index=times)

    for m in methods:
        df_comparison[f"power_{m}"] = results[m]["power"]

    df_comparison["ghi"] = weather_data["ghi"]
    df_comparison["cloud_cover"] = weather_data["cloud_cover"]
    df_comparison["temperature"] = weather_data["temperature"]
    df_comparison["is_day"] = weather_data["is_day"]
    
    # Calculate solar zenith for daylight filtering
    solar_pos = estimator.calculate_solar_position(times)
    df_comparison["zenith"] = solar_pos["zenith"]
    
    # Filter for daylight hours if not showing night
    if not show_night:
        df_daylight = df_comparison[df_comparison["zenith"] < 90].copy()
    else:
        df_daylight = df_comparison.copy()

    # Hourly energy (kWh)
    for m in methods:
        df_comparison[f"energy_{m}"] = df_comparison[f"power_{m}"] / 1000.0
        df_daylight[f"energy_{m}"] = df_daylight[f"power_{m}"] / 1000.0

    # Simulate battery operation
    solar_power_kw = results["combined"]["power"] / 1000  # Convert to kW
    battery_results = estimator.simulate_battery_operation(
        solar_power_kw=solar_power_kw,
        baseload_kw=baseload,
        battery_capacity_kwh=battery_capacity,
        initial_soc_percent=initial_soc,
        max_charge_rate_kw=max_charge_rate,
        max_discharge_rate_kw=max_discharge_rate
    )
    
    # Add battery results to dataframe
    df_comparison["battery_soc_percent"] = battery_results["battery_soc_percent"]
    df_comparison["grid_import_kw"] = battery_results["grid_import_kw"]
    df_comparison["grid_export_kw"] = battery_results["grid_export_kw"]
    df_comparison["self_consumption_kw"] = battery_results["self_consumption_kw"]
    df_comparison["net_load_kw"] = baseload - solar_power_kw

    # Summary statistics
    summary_data = []
    for m in methods:
        peak_kw = df_comparison[f"power_{m}"].max() / 1000
        energy_kwh_today = df_comparison[f"energy_{m}"].resample("D").sum().iloc[0]
        summary_data.append(
            {
                "Method": m.upper(),
                "Peak (kW)": round(peak_kw, 2),
                "Today (kWh)": round(energy_kwh_today, 1),
                "Diff from Max (%)": round(
                    peak_kw / max_calc["total_max_kw"] * 100, 1
                ),
            }
        )
    summary_df = pd.DataFrame(summary_data)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📈 Method comparison")
        st.dataframe(summary_df, use_container_width=True)

    with col2:
        st.markdown("### ⚡ System performance")
        st.metric("System capacity (kWp)", f"{TOTAL_KWP:.2f}")
        st.metric("Theoretical max (kW)", f"{max_calc['total_max_kw']:.2f}")
        system_efficiency = LIVE_PEAK_KW / max_calc["total_max_kw"] * 100
        st.metric("Current efficiency (%)", f"{system_efficiency:.1f}")
        
        # Battery metrics
        final_soc = battery_results["battery_soc_percent"][-1]
        st.metric("Final Battery SOC", f"{final_soc:.1f}%")

    with col3:
        st.markdown("### 🔋 Energy Balance")
        st.metric("Self-sufficiency", f"{battery_results['self_sufficiency_percent']:.1f}%")
        st.metric("Total Grid Import", f"{battery_results['total_grid_import_kwh']:.1f} kWh")
        st.metric("Total Grid Export", f"{battery_results['total_grid_export_kwh']:.1f} kWh")
        st.metric("Solar to Battery", f"{battery_results['solar_to_battery_kwh']:.1f} kWh")

    # --- Sunrise/Sunset Information ---
    st.markdown("### 🌅 Sunrise/Sunset Times")
    sunrise_cols = st.columns(min(len(sunrise_sunset), 4))
    for idx, (date_val, info) in enumerate(list(sunrise_sunset.items())[:4]):
        with sunrise_cols[idx % 4]:
            st.metric(
                f"{date_val}",
                f"{info['sunrise'].strftime('%H:%M')} - {info['sunset'].strftime('%H:%M')}",
                f"{info['daylight_hours']}h daylight"
            )

    # --- PLOTS ---
    st.markdown("### 📊 Power, Energy and Battery Analysis")

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(
        f"Enhanced Solar Production with Battery Storage - {datetime.now().strftime('%d %b %Y')}",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Solar Production and Baseload
    ax1 = axes[0, 0]
    for m in methods:
        ax1.plot(
            df_daylight.index,
            df_daylight[f"power_{m}"] / 1000,
            label=m.upper(),
            alpha=0.8,
            linewidth=1.5,
        )
    ax1.axhline(baseload, color='purple', linestyle='--', linewidth=2, label=f'Baseload ({baseload}kW)')
    ax1.axhline(LIVE_PEAK_KW, color="red", linestyle="--", label="Your Live Peak")
    ax1.fill_between(df_daylight.index, 0, baseload, alpha=0.1, color='purple')
    ax1.set_ylabel("Power (kW)", fontweight="bold")
    ax1.set_title("Solar Production vs Baseload")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Shade night hours if showing them
    if show_night:
        night_mask = df_comparison["zenith"] >= 90
        for night_start in df_comparison.index[night_mask]:
            ax1.axvspan(night_start, night_start + timedelta(hours=1), 
                       alpha=0.1, color='gray')

    # Plot 2: Battery State of Charge
    ax2 = axes[0, 1]
    ax2.plot(df_comparison.index, df_comparison["battery_soc_percent"], 
             color='green', linewidth=2, label='Battery SOC')
    ax2.fill_between(df_comparison.index, 0, df_comparison["battery_soc_percent"], 
                     alpha=0.3, color='green')
    ax2.set_ylabel("Battery SOC (%)", fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.set_title("Battery State of Charge")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Grid Import/Export
    ax3 = axes[1, 0]
    ax3.bar(df_comparison.index, df_comparison["grid_import_kw"], 
            width=0.03, color='red', alpha=0.6, label='Grid Import')
    ax3.bar(df_comparison.index, -df_comparison["grid_export_kw"], 
            width=0.03, color='green', alpha=0.6, label='Grid Export')
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.set_ylabel("Grid Power (kW)", fontweight="bold")
    ax3.set_title("Grid Import/Export")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Self-Consumption Analysis
    ax4 = axes[1, 1]
    solar_kw = solar_power_kw
    self_cons = df_comparison["self_consumption_kw"]
    export = df_comparison["grid_export_kw"]
    
    ax4.stackplot(df_comparison.index, 
                  self_cons, 
                  export,
                  labels=['Self-Consumed', 'Exported'],
                  colors=['blue', 'orange'],
                  alpha=0.7)
    ax4.plot(df_comparison.index, solar_kw, 'k--', linewidth=1, label='Total Solar')
    ax4.set_ylabel("Power (kW)", fontweight="bold")
    ax4.set_title("Solar Self-Consumption Analysis")
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Net Load Profile
    ax5 = axes[2, 0]
    net_load = df_comparison["net_load_kw"]
    positive_mask = net_load >= 0
    negative_mask = net_load < 0
    
    ax5.bar(df_comparison.index[positive_mask], net_load[positive_mask], 
            width=0.03, color='red', alpha=0.6, label='Grid Needed')
    ax5.bar(df_comparison.index[negative_mask], net_load[negative_mask], 
            width=0.03, color='green', alpha=0.6, label='Excess Solar')
    ax5.axhline(0, color='black', linewidth=0.5)
    ax5.set_ylabel("Net Load (kW)", fontweight="bold")
    ax5.set_xlabel("Time")
    ax5.set_title("Net Load Profile (Baseload - Solar)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Performance ratio vs theoretical max
    ax6 = axes[2, 1]
    for m in methods:
        performance_ratio = (
            df_daylight[f"power_{m}"] / 1000 / max_calc["total_max_kw"] * 100
        )
        ax6.plot(
            df_daylight.index,
            performance_ratio,
            label=m.upper(),
            alpha=0.7,
            linewidth=1.5,
        )
    ax6.set_ylabel("Performance Ratio (%)", fontweight="bold")
    ax6.set_xlabel("Time")
    ax6.set_title("Performance Ratio vs Theoretical Max (Daylight Only)")
    ax6.legend(loc="upper right", fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 100)

    plt.tight_layout()
    st.pyplot(fig)

    # --- DAILY SUMMARY ---
    st.markdown("### 📅 Daily Energy Summary")
    
    # Create daily summary
    daily_summary = pd.DataFrame()
    for m in methods:
        daily_summary[f'{m}_kwh'] = df_comparison[f'energy_{m}'].resample('D').sum()
    
    daily_summary['grid_import_kwh'] = df_comparison['grid_import_kw'].resample('D').sum()
    daily_summary['grid_export_kwh'] = df_comparison['grid_export_kw'].resample('D').sum()
    daily_summary['self_consumption_kwh'] = df_comparison['self_consumption_kw'].resample('D').sum()
    
    # Calculate self-sufficiency for each day
    daily_summary['self_sufficiency_%'] = (daily_summary['self_consumption_kwh'] / 
                                          (daily_summary['combined_kwh'] + 1e-6) * 100)
    
    st.dataframe(daily_summary.round(1), use_container_width=True)

    # --- RECOMMENDATIONS ---
    st.markdown("### 💡 Recommendations")
    
    # Find best charging times
    excess_solar_mask = solar_power_kw > baseload
    if excess_solar_mask.any():
        best_charge_hours = pd.Series(solar_power_kw - baseload)[excess_solar_mask]
        best_charge_times = best_charge_hours.nlargest(3).index
        
        st.write("**Best times for battery charging:**")
        for time in best_charge_times:
            excess = solar_power_kw[time] - baseload
            st.write(f"- {time.strftime('%H:%M')}: {excess:.2f} kW excess solar available")
    
    # Grid import analysis
    grid_import_hours = df_comparison[df_comparison['grid_import_kw'] > 0]
    if len(grid_import_hours) > 0:
        st.write("**Grid import occurs during:**")
        for idx, row in grid_import_hours.head(3).iterrows():
            st.write(f"- {idx.strftime('%H:%M')}: {row['grid_import_kw']:.2f} kW from grid")

    # Export results
    results_summary = {
        "system": {
            "total_kwp": TOTAL_KWP,
            "battery_capacity_kwh": float(battery_capacity),
            "baseload_kw": float(baseload),
        },
        "battery_performance": {
            "final_soc_percent": float(battery_results["battery_soc_percent"][-1]),
            "self_sufficiency_percent": float(battery_results["self_sufficiency_percent"]),
            "total_grid_import_kwh": float(battery_results["total_grid_import_kwh"]),
            "total_grid_export_kwh": float(battery_results["total_grid_export_kwh"]),
        },
        "predictions": {
            m: {
                "peak_kw": float(df_comparison[f"power_{m}"].max() / 1000),
                "today_kwh": float(
                    df_comparison[f"energy_{m}"].resample("D").sum().iloc[0]
                ),
            }
            for m in methods
        },
        "daily_summary": daily_summary.to_dict(orient="index"),
        "sunrise_sunset": {str(k): {"sunrise": v["sunrise"].strftime("%H:%M"), 
                                   "sunset": v["sunset"].strftime("%H:%M")} 
                          for k, v in sunrise_sunset.items()}
    }

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download JSON results",
            data=json.dumps(results_summary, indent=2),
            file_name=f"solar_battery_analysis_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
        )
    
    with col2:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()


if __name__ == "__main__":
    run_app()
