import json
import warnings
from datetime import datetime, date, timedelta, time
from typing import Dict, List, Tuple

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
MIN_SOLAR_THRESHOLD = 20  # W/m² minimum for production

# Tariff defaults
EXPORT_TARIFF_GBP_PER_KWH = 0.15  # 15p/kWh
LOW_TARIFF_START = time(23, 30)  # 23:30
LOW_TARIFF_END = time(5, 30)    # 05:30
PEAK_FACTOR = 2.0  # Peak hours multiplier

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

            # Calculate GHI - ensure night hours are zero
            ghi = np.maximum(0, direct_rad + diffuse_rad)
            # Set GHI to zero for night hours
            ghi = ghi * is_day

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
            return _self._generate_simulated_data(days)

    def _generate_simulated_data(self, days):
        """Generate realistic simulated data"""
        n_points = days * 24
        now = datetime.now()
        start_time = datetime(now.year, now.month, now.day, now.hour)
        times = pd.date_range(start=start_time, periods=n_points, freq="H")

        # Generate day/night cycle based on solar position
        solar_pos = self.calculate_solar_position(times)
        zenith = solar_pos["zenith"]
        is_day = zenith < 90

        t = np.linspace(0, days * 2 * np.pi, n_points)
        
        # Solar pattern based on time of day
        hour_of_day = times.hour.values
        day_pattern = np.zeros_like(hour_of_day, dtype=float)
        for hour in range(24):
            hour_mask = hour_of_day == hour
            if 6 <= hour <= 18:
                intensity = np.sin(np.pi * (hour - 6) / 12) ** 2
                day_pattern[hour_mask] = intensity
        
        if days > 1:
            day_pattern = np.tile(day_pattern[:24], days)[:n_points]

        cloud_noise = 0.3 * np.sin(t / 3) + 0.2 * np.sin(t / 7) + 0.1 * np.random.randn(n_points)
        cloud_cover = np.clip(30 + 40 * (cloud_noise + 1) / 2, 0, 100)

        ghi_clear = 1000 * day_pattern * np.clip(1 - 0.1 * np.sin(t / 10), 0.7, 1)
        cloud_factor = 1 - 0.8 * (cloud_cover / 100) ** 1.5
        ghi = ghi_clear * cloud_factor
        ghi = np.where(is_day, ghi, 0)

        temperature = 15 + 10 * day_pattern + 5 * np.sin(t / 10)

        return {
            "times": times,
            "ghi": ghi,
            "dni": np.where(is_day, ghi * 0.7, 0),
            "dhi": np.where(is_day, ghi * 0.3, 0),
            "gti": np.where(is_day, ghi * 1.1, 0),
            "cloud_cover": cloud_cover,
            "temperature": temperature,
            "humidity": 60 + 20 * np.sin(t / 5),
            "wind_speed": 2 + 3 * day_pattern,
            "pressure": 1013 * np.ones_like(ghi),
            "direct_rad": np.where(is_day, ghi * 0.6, 0),
            "diffuse_rad": np.where(is_day, ghi * 0.4, 0),
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

    def pvwatts_model(self, ghi, temperature, wind_speed, solar_pos):
        T_ref = 25
        G_ref = 1000
        gamma = -0.004
        
        zenith = solar_pos["zenith"]
        
        # Ensure zero production at night
        mask = (zenith < 90) & (ghi > MIN_SOLAR_THRESHOLD)
        
        T_module = temperature + ghi * np.exp(-3.47 - 0.0594 * wind_speed) / 1000

        power_dc = (
            self.total_kwp
            * 1000
            * (ghi / G_ref)
            * (1 + gamma * (T_module - T_ref))
        )
        power_dc = np.where(mask, power_dc, 0)

        power_dc_per_unit = np.zeros_like(power_dc)
        non_zero_mask = power_dc > 0
        if np.any(non_zero_mask):
            power_dc_per_unit[non_zero_mask] = power_dc[non_zero_mask] / (self.total_kwp * 1000 + 1e-6)
        
        inv_eff = 0.96 - 0.05 * power_dc_per_unit + 0.04 * power_dc_per_unit ** 2
        inv_eff = np.clip(inv_eff, 0.9, 0.965)

        power_ac = power_dc * inv_eff * SYSTEM_EFFICIENCY
        power_ac = np.clip(power_ac, 0, self.total_kwp * 1000 * 1.1)
        
        # Final zero-check for night
        power_ac = np.where(mask, power_ac, 0)

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
            poa_total = np.where(zenith < 90, poa_total, 0)

            array_kwp = array["n_modules"] * P_MODULE_WP / 1000
            array_power = (
                poa_total / 1000 * array_kwp * array["loss_factor"] * 0.96
            )

            total_poa += array_power * 1000

        return total_poa

    def estimate_production(self, weather_data, method="combined"):
        solar_pos = self.calculate_solar_position(weather_data["times"])
        zenith = solar_pos["zenith"]
        
        # Day mask - only produce when sun is above horizon
        day_mask = zenith < 90

        p_pvwatts, T_module, inv_eff = self.pvwatts_model(
            weather_data["ghi"],
            weather_data["temperature"],
            weather_data["wind_speed"],
            solar_pos,
        )

        p_perez = self.perez_model_poa(weather_data, solar_pos)

        if method == "pvwatts":
            final_power = p_pvwatts
        elif method == "perez":
            final_power = p_perez
        elif method == "combined":
            # Weighted combination
            final_power = 0.5 * p_pvwatts + 0.5 * p_perez
        else:
            final_power = p_pvwatts
        
        # Final zero-check for night
        final_power = np.where(day_mask, final_power, 0)

        return {
            "power": final_power,
            "pvwatts": np.where(day_mask, p_pvwatts, 0),
            "perez": np.where(day_mask, p_perez, 0),
            "temperature": T_module,
            "inverter_eff": inv_eff,
            "solar_zenith": solar_pos["zenith"],
            "day_mask": day_mask,
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
        }

        solar_pos = self.calculate_solar_position(times)
        
        # Clear sky model simplified
        zenith = solar_pos["zenith"]
        day_mask = zenith < 90
        solar_noon_mask = (zenith == zenith.min())  # Approximate solar noon
        
        # Create ideal GHI profile
        ideal_ghi = np.zeros(len(times))
        for i in range(len(times)):
            if day_mask[i]:
                # Parabolic profile peaking at solar noon
                hour_angle = np.abs(solar_pos["hour_angle"][i])
                ideal_ghi[i] = 1000 * np.cos(np.radians(hour_angle * 0.75))
        
        ideal_weather["ghi"] = ideal_ghi
        ideal_weather["dni"] = ideal_ghi * 0.7
        ideal_weather["dhi"] = ideal_ghi * 0.3

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

    def simulate_battery_self_powered(self, solar_power_kw, baseload_kw, 
                                    battery_capacity_kwh, initial_soc_percent=50,
                                    max_charge_rate_kw=5.0, max_discharge_rate_kw=5.0,
                                    reserve_percent=0):
        """
        Simulate Tesla Powerwall Self-Powered mode
        Maximizes self-consumption, doesn't charge from grid
        """
        n = len(solar_power_kw)
        battery_soc = np.zeros(n)
        grid_import = np.zeros(n)
        grid_export = np.zeros(n)
        self_consumption = np.zeros(n)
        
        # Convert initial SOC to kWh
        current_soc_kwh = battery_capacity_kwh * initial_soc_percent / 100
        reserve_kwh = battery_capacity_kwh * reserve_percent / 100
        
        for i in range(n):
            solar_gen = solar_power_kw[i]
            net_load = baseload_kw - solar_gen
            
            if net_load < 0:  # Excess solar
                excess = -net_load
                
                # Charge battery with excess solar (up to max charge rate)
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
                
                # Try to discharge battery (but keep reserve)
                discharge_possible = min(
                    deficit,
                    max_discharge_rate_kw,
                    (current_soc_kwh - reserve_kwh) / 1.0  # Don't discharge below reserve
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

    def simulate_battery_time_based(self, solar_power_kw, baseload_kw, 
                                  battery_capacity_kwh, initial_soc_percent=50,
                                  max_charge_rate_kw=5.0, max_discharge_rate_kw=5.0,
                                  reserve_percent=10, low_tariff_start=LOW_TARIFF_START,
                                  low_tariff_end=LOW_TARIFF_END, times=None):
        """
        Simulate Tesla Powerwall Time-Based Control mode
        Charges from grid during low tariff, optimizes for next day's solar
        """
        n = len(solar_power_kw)
        battery_soc = np.zeros(n)
        grid_import = np.zeros(n)
        grid_export = np.zeros(n)
        self_consumption = np.zeros(n)
        grid_charge_kw = np.zeros(n)
        
        # Convert initial SOC to kWh
        current_soc_kwh = battery_capacity_kwh * initial_soc_percent / 100
        reserve_kwh = battery_capacity_kwh * reserve_percent / 100
        
        # Create time masks
        time_of_day = pd.Series(times).dt.time
        low_tariff_mask = ((time_of_day >= low_tariff_start) | (time_of_day <= low_tariff_end)).values
        
        # Calculate expected solar for next day to determine overnight charge target
        # Group by date
        df_solar = pd.DataFrame({
            'time': times,
            'solar_kw': solar_power_kw,
            'date': pd.Series(times).dt.date
        })
        
        # Calculate daily solar generation
        daily_solar = df_solar.groupby('date')['solar_kw'].sum().reset_index()
        daily_solar.columns = ['date', 'daily_solar_kwh']
        
        # Create a mapping of date to next day's expected solar
        date_to_solar = dict(zip(daily_solar['date'], daily_solar['daily_solar_kwh']))
        
        # Calculate optimal overnight charge target
        # Target = Expected house demand until next low tariff - Expected solar
        # House demand = baseload * hours + peak_factor * baseload * peak_hours
        hours_until_next_night = 24  # Simplification
        expected_demand = baseload_kw * hours_until_next_night
        
        for i in range(n):
            solar_gen = solar_power_kw[i]
            current_time = times[i]
            current_date = current_time.date()
            
            # Get expected solar for next day
            next_date = current_date + timedelta(days=1)
            expected_next_day_solar = date_to_solar.get(next_date, 0)
            
            # Calculate target SOC for overnight charging
            # We want enough battery to cover expected demand minus expected solar
            target_energy_kwh = max(0, expected_demand - expected_next_day_solar)
            target_energy_kwh = min(target_energy_kwh, battery_capacity_kwh)
            target_soc_percent = (target_energy_kwh / battery_capacity_kwh) * 100
            
            net_load = baseload_kw - solar_gen
            
            if low_tariff_mask[i]:
                # Low tariff period - can charge from grid
                if net_load < 0:  # Excess solar
                    excess = -net_load
                    
                    # Charge battery with excess solar first
                    charge_from_solar = min(
                        excess,
                        max_charge_rate_kw,
                        (battery_capacity_kwh - current_soc_kwh) / 1.0
                    )
                    charge_energy_solar = charge_from_solar * BATTERY_EFFICIENCY
                    current_soc_kwh += charge_energy_solar
                    excess -= charge_from_solar
                    
                    # Then charge from grid to reach target if needed
                    charge_needed = max(0, target_energy_kwh - current_soc_kwh)
                    charge_from_grid = min(
                        charge_needed,
                        max_charge_rate_kw - charge_from_solar,
                        (battery_capacity_kwh - current_soc_kwh) / 1.0
                    )
                    grid_charge_kw[i] = charge_from_grid
                    current_soc_kwh += charge_from_grid
                    
                    # Any remaining excess goes to grid
                    grid_export[i] = excess
                    grid_import[i] = 0
                    self_consumption[i] = solar_gen
                    
                else:  # Solar deficit during low tariff
                    deficit = net_load
                    
                    # Try to discharge battery (but keep reserve)
                    discharge_possible = min(
                        deficit,
                        max_discharge_rate_kw,
                        (current_soc_kwh - reserve_kwh) / 1.0
                    )
                    
                    if discharge_possible > 0:
                        current_soc_kwh -= discharge_possible
                        deficit -= discharge_possible
                        self_consumption[i] = solar_gen
                    else:
                        self_consumption[i] = solar_gen
                    
                    # Import deficit from grid
                    grid_import[i] = deficit
                    grid_export[i] = 0
                    
                    # Also charge from grid to target if needed
                    charge_needed = max(0, target_energy_kwh - current_soc_kwh)
                    charge_from_grid = min(
                        charge_needed,
                        max_charge_rate_kw,
                        (battery_capacity_kwh - current_soc_kwh) / 1.0
                    )
                    grid_charge_kw[i] = charge_from_grid
                    current_soc_kwh += charge_from_grid
                    grid_import[i] += charge_from_grid
                    
            else:
                # High tariff period - minimize grid import
                if net_load < 0:  # Excess solar
                    excess = -net_load
                    
                    # Charge battery with excess solar
                    charge_possible = min(
                        excess,
                        max_charge_rate_kw,
                        (battery_capacity_kwh - current_soc_kwh) / 1.0
                    )
                    charge_energy = charge_possible * BATTERY_EFFICIENCY
                    current_soc_kwh += charge_energy
                    self_consumption[i] = solar_gen
                    
                    # Remaining excess goes to grid
                    grid_export[i] = excess - charge_possible
                    grid_import[i] = 0
                    
                else:  # Solar deficit during high tariff
                    deficit = net_load
                    
                    # Try to discharge battery (but keep reserve)
                    discharge_possible = min(
                        deficit,
                        max_discharge_rate_kw,
                        (current_soc_kwh - reserve_kwh) / 1.0
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
        
        total_export_revenue = np.sum(grid_export) * EXPORT_TARIFF_GBP_PER_KWH
        
        return {
            'battery_soc_kwh': battery_soc,
            'battery_soc_percent': battery_soc / battery_capacity_kwh * 100,
            'grid_import_kw': grid_import,
            'grid_export_kw': grid_export,
            'grid_charge_kw': grid_charge_kw,
            'self_consumption_kw': self_consumption,
            'solar_to_battery_kwh': np.sum(np.maximum(0, solar_power_kw - baseload_kw)),
            'total_grid_import_kwh': np.sum(grid_import),
            'total_grid_export_kwh': np.sum(grid_export),
            'total_grid_charge_kwh': np.sum(grid_charge_kw),
            'total_export_revenue_gbp': total_export_revenue,
            'self_sufficiency_percent': np.sum(self_consumption) / (np.sum(solar_power_kw) + 1e-6) * 100,
            'overnight_charge_target_percent': target_soc_percent
        }

    def calculate_export_potential(self, solar_power_kw, baseload_kw, peak_factor=2.0):
        """
        Calculate potential export power and energy
        Includes consideration of peak house demand
        """
        n = len(solar_power_kw)
        
        # Calculate net export potential
        # During peak hours, house demand = baseload * peak_factor
        hour_of_day = np.arange(n) % 24
        peak_hours = (hour_of_day >= 16) & (hour_of_day <= 19)  # 4-7 PM typical peak
        
        house_demand = np.ones(n) * baseload_kw
        house_demand[peak_hours] = baseload_kw * peak_factor
        
        # Calculate net export
        net_export = np.maximum(0, solar_power_kw - house_demand)
        
        # Calculate statistics
        max_export_kw = np.max(net_export)
        total_export_kwh = np.sum(net_export)
        export_revenue = total_export_kwh * EXPORT_TARIFF_GBP_PER_KWH
        
        return {
            'export_power_kw': net_export,
            'max_export_kw': max_export_kw,
            'total_export_kwh': total_export_kwh,
            'export_revenue_gbp': export_revenue,
            'export_hours': np.sum(net_export > 0),
            'house_demand_kw': house_demand,
            'peak_hours_mask': peak_hours
        }


def run_app():
    st.set_page_config(
        page_title="Enhanced Solar Production Estimator",
        layout="wide",
    )

    st.title("🏠 Enhanced Solar Production with Tesla Powerwall")

    with st.sidebar:
        st.header("⚙️ Configuration")
        lat = st.number_input("Latitude", value=LAT, format="%.4f")
        lon = st.number_input("Longitude", value=LON, format="%.4f")
        
        st.subheader("🔋 Tesla Powerwall Settings")
        
        # Powerwall mode selection
        powerwall_mode = st.radio(
            "Powerwall Mode",
            ["Self-Powered", "Time-Based Control"],
            index=1,
            help="Self-Powered: Maximize self-consumption. Time-Based Control: Charge from grid during low tariff."
        )
        
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
        
        if powerwall_mode == "Time-Based Control":
            reserve_percent = st.slider(
                "Reserve Level (%)",
                0, 50, 10,
                help="Minimum battery level that won't be discharged"
            )
            
            st.subheader("⏰ Time-Based Control Settings")
            
            col1, col2 = st.columns(2)
            with col1:
                low_start_hour = st.number_input("Low Tariff Start Hour", 0, 23, 23)
                low_start_min = st.number_input("Low Start Minute", 0, 59, 30)
            with col2:
                low_end_hour = st.number_input("Low Tariff End Hour", 0, 23, 5)
                low_end_min = st.number_input("Low End Minute", 0, 59, 30)
            
            low_tariff_start = time(low_start_hour, low_start_min)
            low_tariff_end = time(low_end_hour, low_end_min)
            
            st.info(f"Low tariff period: {low_tariff_start.strftime('%H:%M')} to {low_tariff_end.strftime('%H:%M')}")
        
        st.subheader("🏠 House Load Settings")
        baseload = st.number_input(
            "Baseload (kW)", 
            value=BASELOAD_KW,
            min_value=0.0,
            max_value=10.0,
            step=0.1,
            help="Continuous house power consumption"
        )
        
        peak_factor = st.number_input(
            "Peak Load Factor", 
            value=PEAK_FACTOR,
            min_value=1.0,
            max_value=5.0,
            step=0.1,
            help="Multiplier for baseload during peak hours (4-7 PM)"
        )
        
        st.subheader("💰 Export Settings")
        export_tariff = st.number_input(
            "Export Tariff (£/kWh)",
            value=EXPORT_TARIFF_GBP_PER_KWH,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            format="%.3f",
            help="Price you receive for exported electricity"
        )
        
        st.subheader("📈 Forecast Settings")
        days = st.slider("Forecast days", min_value=1, max_value=7, value=3)
        show_night = st.checkbox("Show night hours", value=False)
        
        method = st.selectbox(
            "Solar estimation method",
            options=["combined", "pvwatts", "perez"],
            index=0,
        )

    estimator = EnhancedSolarEstimator(lat, lon, arrays)

    st.subheader("📊 Weather data and predictions")
    with st.spinner("Fetching weather data and running models..."):
        weather_data = estimator.get_comprehensive_weather_data(days=days)
        max_calc = estimator.calculate_expected_max(days=days)

        methods = ["pvwatts", "perez", "combined"]
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
    df_comparison["zenith"] = results["combined"]["solar_zenith"]
    df_comparison["day_mask"] = results["combined"]["day_mask"]
    
    # Filter for daylight hours if not showing night
    if not show_night:
        df_daylight = df_comparison[df_comparison["day_mask"]].copy()
    else:
        df_daylight = df_comparison.copy()

    # Hourly energy (kWh)
    for m in methods:
        df_comparison[f"energy_{m}"] = df_comparison[f"power_{m}"] / 1000.0
        df_daylight[f"energy_{m}"] = df_daylight[f"power_{m}"] / 1000.0

    # Get solar power for battery simulation
    solar_power_kw = results["combined"]["power"] / 1000  # Convert to kW
    
    # Calculate export potential
    export_results = estimator.calculate_export_potential(
        solar_power_kw=solar_power_kw,
        baseload_kw=baseload,
        peak_factor=peak_factor
    )
    
    # Run battery simulation based on mode
    if powerwall_mode == "Self-Powered":
        battery_results = estimator.simulate_battery_self_powered(
            solar_power_kw=solar_power_kw,
            baseload_kw=baseload,
            battery_capacity_kwh=battery_capacity,
            initial_soc_percent=initial_soc,
            max_charge_rate_kw=max_charge_rate,
            max_discharge_rate_kw=max_discharge_rate,
            reserve_percent=0
        )
    else:  # Time-Based Control
        battery_results = estimator.simulate_battery_time_based(
            solar_power_kw=solar_power_kw,
            baseload_kw=baseload,
            battery_capacity_kwh=battery_capacity,
            initial_soc_percent=initial_soc,
            max_charge_rate_kw=max_charge_rate,
            max_discharge_rate_kw=max_discharge_rate,
            reserve_percent=reserve_percent,
            low_tariff_start=low_tariff_start,
            low_tariff_end=low_tariff_end,
            times=times
        )
    
    # Add battery and export results to dataframe
    df_comparison["battery_soc_percent"] = battery_results["battery_soc_percent"]
    df_comparison["grid_import_kw"] = battery_results["grid_import_kw"]
    df_comparison["grid_export_kw"] = battery_results["grid_export_kw"]
    df_comparison["self_consumption_kw"] = battery_results["self_consumption_kw"]
    df_comparison["export_potential_kw"] = export_results['export_power_kw']
    df_comparison["house_demand_kw"] = export_results['house_demand_kw']
    df_comparison["peak_hours"] = export_results['peak_hours_mask']
    df_comparison["net_load_kw"] = baseload - solar_power_kw
    
    if powerwall_mode == "Time-Based Control":
        df_comparison["grid_charge_kw"] = battery_results["grid_charge_kw"]
        df_comparison["low_tariff"] = (
            (pd.Series(times).dt.time >= low_tariff_start) | 
            (pd.Series(times).dt.time <= low_tariff_end)
        ).values

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

    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📈 Solar Production")
        st.metric("System capacity (kWp)", f"{TOTAL_KWP:.2f}")
        st.metric("Theoretical max (kW)", f"{max_calc['total_max_kw']:.2f}")
        st.metric("Today's estimated solar", f"{summary_df.loc[summary_df['Method'] == 'COMBINED', 'Today (kWh)'].iloc[0]:.1f} kWh")

    with col2:
        st.markdown("### 🔋 Battery Performance")
        final_soc = battery_results["battery_soc_percent"][-1]
        st.metric("Final SOC", f"{final_soc:.1f}%")
        st.metric("Self-sufficiency", f"{battery_results['self_sufficiency_percent']:.1f}%")
        if powerwall_mode == "Time-Based Control":
            st.metric("Overnight charge target", f"{battery_results.get('overnight_charge_target_percent', 0):.0f}%")

    with col3:
        st.markdown("### 💰 Financials")
        st.metric("Export revenue", f"£{export_results['export_revenue_gbp']:.2f}")
        st.metric("Total export", f"{export_results['total_export_kwh']:.1f} kWh")
        st.metric("Max export power", f"{export_results['max_export_kw']:.2f} kW")

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
        f"Tesla Powerwall: {powerwall_mode} Mode - {datetime.now().strftime('%d %b %Y')}",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Solar Production and House Demand
    ax1 = axes[0, 0]
    # Plot solar production
    ax1.plot(
        df_daylight.index,
        solar_power_kw[df_comparison["day_mask"]],
        color='gold',
        linewidth=2,
        label='Solar Production'
    )
    # Plot house demand
    ax1.fill_between(
        df_comparison.index,
        0,
        df_comparison["house_demand_kw"],
        alpha=0.3,
        color='red',
        label='House Demand'
    )
    # Highlight peak hours
    peak_mask = df_comparison["peak_hours"]
    if peak_mask.any():
        ax1.fill_between(
            df_comparison.index[peak_mask],
            0,
            df_comparison["house_demand_kw"][peak_mask],
            alpha=0.5,
            color='darkred',
            label='Peak Hours'
        )
    
    ax1.set_ylabel("Power (kW)", fontweight="bold")
    ax1.set_title("Solar Production vs House Demand")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Battery State of Charge
    ax2 = axes[0, 1]
    ax2.plot(df_comparison.index, df_comparison["battery_soc_percent"], 
             color='green', linewidth=2, label='Battery SOC')
    ax2.fill_between(df_comparison.index, 0, df_comparison["battery_soc_percent"], 
                     alpha=0.3, color='green')
    
    # Add reserve level line for Time-Based Control
    if powerwall_mode == "Time-Based Control":
        ax2.axhline(y=reserve_percent, color='red', linestyle='--', alpha=0.7, 
                   label=f'Reserve ({reserve_percent}%)')
    
    ax2.set_ylabel("Battery SOC (%)", fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.set_title("Battery State of Charge")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Grid Import/Export
    ax3 = axes[1, 0]
    # Stack import and export
    import_mask = df_comparison["grid_import_kw"] > 0
    export_mask = df_comparison["grid_export_kw"] > 0
    
    if import_mask.any():
        ax3.bar(df_comparison.index[import_mask], df_comparison["grid_import_kw"][import_mask], 
                width=0.03, color='red', alpha=0.6, label='Grid Import')
    if export_mask.any():
        ax3.bar(df_comparison.index[export_mask], -df_comparison["grid_export_kw"][export_mask], 
                width=0.03, color='green', alpha=0.6, label='Grid Export')
    
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.set_ylabel("Grid Power (kW)", fontweight="bold")
    ax3.set_title("Grid Import/Export")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Export Potential
    ax4 = axes[1, 1]
    ax4.plot(df_comparison.index, df_comparison["export_potential_kw"], 
             color='orange', linewidth=2, label='Export Potential')
    ax4.fill_between(df_comparison.index, 0, df_comparison["export_potential_kw"], 
                     alpha=0.3, color='orange')
    
    # Add cumulative export value
    ax4_twin = ax4.twinx()
    cumulative_export = np.cumsum(df_comparison["export_potential_kw"]) * export_tariff
    ax4_twin.plot(df_comparison.index, cumulative_export, 
                  color='purple', linestyle='--', label='Cumulative Export Value')
    ax4_twin.set_ylabel("Export Value (£)", color='purple')
    ax4_twin.tick_params(axis='y', labelcolor='purple')
    
    ax4.set_ylabel("Export Power (kW)", fontweight="bold")
    ax4.set_title("Export Potential and Value")
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Time-Based Control Features (if applicable)
    ax5 = axes[2, 0]
    if powerwall_mode == "Time-Based Control":
        # Plot low tariff periods
        low_tariff_mask = df_comparison.get("low_tariff", np.zeros(len(df_comparison), dtype=bool))
        if low_tariff_mask.any():
            for start_time in df_comparison.index[low_tariff_mask]:
                ax5.axvspan(start_time, start_time + timedelta(hours=1), 
                           alpha=0.2, color='blue', label='Low Tariff' if start_time == df_comparison.index[low_tariff_mask][0] else "")
        
        # Plot grid charging
        grid_charge_mask = df_comparison.get("grid_charge_kw", np.zeros(len(df_comparison))) > 0
        if grid_charge_mask.any():
            ax5.bar(df_comparison.index[grid_charge_mask], 
                   df_comparison["grid_charge_kw"][grid_charge_mask], 
                   width=0.03, color='cyan', alpha=0.7, label='Grid Charging')
        
        ax5.set_ylabel("Power (kW)", fontweight="bold")
        ax5.set_title("Time-Based Control: Low Tariff & Grid Charging")
        ax5.legend()
    else:
        # Plot self-consumption for Self-Powered mode
        ax5.plot(df_comparison.index, df_comparison["self_consumption_kw"], 
                color='blue', linewidth=2, label='Self-Consumption')
        ax5.fill_between(df_comparison.index, 0, df_comparison["self_consumption_kw"], 
                        alpha=0.3, color='blue')
        ax5.set_ylabel("Power (kW)", fontweight="bold")
        ax5.set_title("Self-Consumption")
        ax5.legend()
    
    ax5.grid(True, alpha=0.3)

    # Plot 6: Net Load Profile
    ax6 = axes[2, 1]
    net_load = df_comparison["net_load_kw"]
    positive_mask = (net_load >= 0) & df_comparison["day_mask"]
    negative_mask = (net_load < 0) & df_comparison["day_mask"]
    
    if positive_mask.any():
        ax6.bar(df_comparison.index[positive_mask], net_load[positive_mask], 
                width=0.03, color='red', alpha=0.6, label='Grid Import Needed')
    if negative_mask.any():
        ax6.bar(df_comparison.index[negative_mask], net_load[negative_mask], 
                width=0.03, color='green', alpha=0.6, label='Excess Solar')
    
    ax6.axhline(0, color='black', linewidth=0.5)
    ax6.set_ylabel("Net Load (kW)", fontweight="bold")
    ax6.set_xlabel("Time")
    ax6.set_title("Net Load Profile (House Demand - Solar)")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # --- DAILY SUMMARY ---
    st.markdown("### 📅 Daily Energy Summary")
    
    daily_summary = pd.DataFrame()
    for m in methods:
        daily_summary[f'{m}_kwh'] = df_comparison[f'energy_{m}'].resample('D').sum()
    
    daily_summary['grid_import_kwh'] = df_comparison['grid_import_kw'].resample('D').sum()
    daily_summary['grid_export_kwh'] = df_comparison['grid_export_kw'].resample('D').sum()
    daily_summary['self_consumption_kwh'] = df_comparison['self_consumption_kw'].resample('D').sum()
    daily_summary['export_potential_kwh'] = df_comparison['export_potential_kw'].resample('D').sum()
    
    # Calculate financials
    daily_summary['export_value_gbp'] = daily_summary['export_potential_kwh'] * export_tariff
    daily_summary['self_sufficiency_%'] = (daily_summary['self_consumption_kwh'] / 
                                          (daily_summary['combined_kwh'] + 1e-6) * 100)
    
    st.dataframe(daily_summary.round(1), use_container_width=True)

    # --- RECOMMENDATIONS ---
    st.markdown("### 💡 Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔋 Battery Optimization")
        
        # Find best charging times
        excess_solar_mask = (solar_power_kw > baseload) & df_comparison["day_mask"]
        if excess_solar_mask.any():
            best_charge_hours = pd.Series(solar_power_kw - baseload)[excess_solar_mask]
            best_charge_times = best_charge_hours.nlargest(3).index
            
            st.write("**Best times for solar charging:**")
            for time in best_charge_times:
                excess = solar_power_kw[time] - baseload
                st.write(f"- {time.strftime('%H:%M')}: {excess:.2f} kW excess solar")
        
        if powerwall_mode == "Time-Based Control":
            st.write("**Time-Based Control Strategy:**")
            st.write(f"- Reserve level: {reserve_percent}%")
            st.write(f"- Low tariff charging: {low_tariff_start.strftime('%H:%M')} to {low_tariff_end.strftime('%H:%M')}")
            if 'overnight_charge_target_percent' in battery_results:
                st.write(f"- Target overnight charge: {battery_results['overnight_charge_target_percent']:.0f}%")
    
    with col2:
        st.markdown("#### 💰 Financial Opportunities")
        
        # Export opportunities
        if export_results['export_hours'] > 0:
            st.write("**Export opportunities:**")
            st.write(f"- Total export potential: {export_results['total_export_kwh']:.1f} kWh")
            st.write(f"- Export revenue: £{export_results['export_revenue_gbp']:.2f}")
            st.write(f"- Max export power: {export_results['max_export_kw']:.2f} kW")
            
            # Find best export times
            high_export_mask = df_comparison["export_potential_kw"] > 1.0  # More than 1kW
            if high_export_mask.any():
                export_times = df_comparison.index[high_export_mask]
                st.write("**Best export times today:**")
                for i, time in enumerate(export_times[:3]):  # Show top 3
                    export_power = df_comparison["export_potential_kw"][time]
                    st.write(f"- {time.strftime('%H:%M')}: {export_power:.2f} kW")

    # Export results
    results_summary = {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "total_kwp": TOTAL_KWP,
            "battery_capacity_kwh": float(battery_capacity),
            "baseload_kw": float(baseload),
            "powerwall_mode": powerwall_mode,
        },
        "battery_performance": {
            "final_soc_percent": float(battery_results["battery_soc_percent"][-1]),
            "self_sufficiency_percent": float(battery_results["self_sufficiency_percent"]),
            "total_grid_import_kwh": float(battery_results["total_grid_import_kwh"]),
            "total_grid_export_kwh": float(battery_results["total_grid_export_kwh"]),
        },
        "export_potential": {
            "total_export_kwh": float(export_results['total_export_kwh']),
            "max_export_kw": float(export_results['max_export_kw']),
            "export_revenue_gbp": float(export_results['export_revenue_gbp']),
            "export_tariff_gbp_per_kwh": float(export_tariff),
        },
        "daily_summary": daily_summary.to_dict(orient="index"),
    }

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download JSON results",
            data=json.dumps(results_summary, indent=2),
            file_name=f"powerwall_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
        )
    
    with col2:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()


if __name__ == "__main__":
    run_app()
