import json
import warnings
from datetime import datetime, date

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

    @st.cache_data(show_spinner=False)
    def get_comprehensive_weather_data(_self, days=3):
        """Get comprehensive weather data with multiple parameters"""
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={_self.lat}&longitude={_self.lon}"
            f"&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,"
            f"surface_pressure,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,"
            f"direct_radiation,diffuse_radiation,global_tilted_irradiance,"
            f"direct_normal_irradiance,shortwave_radiation,precipitation,"
            f"wind_speed_10m,wind_direction_10m"
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
            }

        except Exception:
            # Fallback to simulated data if API fails
            return _self._generate_simulated_data(days)

    def _generate_simulated_data(self, days):
        """Generate realistic simulated data"""
        n_points = days * 24
        times = pd.date_range(start=datetime.now(), periods=n_points, freq="H")

        t = np.linspace(0, days * 2 * np.pi, n_points)

        day_pattern = np.sin(np.linspace(0, np.pi, 24)) ** 2
        day_pattern = np.tile(day_pattern, days)

        cloud_noise = 0.3 * np.sin(t / 3) + 0.2 * np.sin(t / 7) + 0.1 * np.random.randn(
            n_points
        )
        cloud_cover = np.clip(30 + 40 * (cloud_noise + 1) / 2, 0, 100)

        ghi_clear = 1000 * day_pattern * np.clip(1 - 0.1 * np.sin(t / 10), 0.7, 1)
        cloud_factor = 1 - 0.8 * (cloud_cover / 100) ** 1.5
        ghi = ghi_clear * cloud_factor

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

        T_module = temperature + ghi * np.exp(-3.47 - 0.0594 * wind_speed) / 1000

        power_dc = (
            self.total_kwp
            * 1000
            * (ghi / G_ref)
            * (1 + gamma * (T_module - T_ref))
        )

        power_dc_per_unit = power_dc / (self.total_kwp * 1000)
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
        times = pd.date_range(start=date.today(), periods=days * 24, freq="H")

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

        # Use combined method on clear-sky GHI to get “max” production
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


def run_app():
    st.set_page_config(
        page_title="Enhanced Solar Production Estimator",
        layout="wide",
    )

    st.title("Enhanced Solar Production Estimation System")

    with st.sidebar:
        st.header("Configuration")
        lat = st.number_input("Latitude", value=LAT, format="%.4f")
        lon = st.number_input("Longitude", value=LON, format="%.4f")
        live_peak_kw = st.number_input(
            "Observed live peak (kW)", value=LIVE_PEAK_KW, format="%.2f"
        )
        days = st.slider("Forecast days", min_value=1, max_value=365, value=120)
        method = st.selectbox(
            "Default method to highlight",
            options=["combined", "pvwatts", "perez", "cloud"],
            index=0,
        )

    estimator = EnhancedSolarEstimator(lat, lon, arrays)

    st.subheader("Weather data and predictions")
    with st.spinner("Fetching weather data and running models..."):
        weather_data = estimator.get_comprehensive_weather_data(days=days)
        max_calc = estimator.calculate_expected_max(days=days)

        methods = ["pvwatts", "perez", "cloud", "combined"]
        results = {
            m: estimator.estimate_production(weather_data, method=m) for m in methods
        }

    # Use full time horizon
    times = weather_data["times"]
    df_comparison = pd.DataFrame(index=times)

    for m in methods:
        df_comparison[f"power_{m}"] = results[m]["power"]

    df_comparison["ghi"] = weather_data["ghi"]
    df_comparison["cloud_cover"] = weather_data["cloud_cover"]
    df_comparison["temperature"] = weather_data["temperature"]

    # Hourly energy (kWh)
    for m in methods:
        df_comparison[f"energy_{m}"] = df_comparison[f"power_{m}"] / 1000.0

    # Summary stats for first day
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

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Method comparison summary")
        st.dataframe(summary_df, use_container_width=True)

    with col2:
        st.markdown("### System performance")
        st.metric("System capacity (kWp)", f"{TOTAL_KWP:.2f}")
        st.metric("Observed live peak (kW)", f"{live_peak_kw:.2f}")
        st.metric(
            "Theoretical max (kW)",
            f"{max_calc['total_max_kw']:.2f}",
        )
        system_efficiency = live_peak_kw / max_calc["total_max_kw"] * 100
        st.metric("Current efficiency (%)", f"{system_efficiency:.1f}")

    # --- DAILY AND MONTHLY AVERAGES ---

    # Regular hourly index for robust resampling
    df_hourly = df_comparison.asfreq("H")

    # Daily total energy per method
    daily_energy = pd.DataFrame(
        {
            m: df_hourly[f"energy_{m}"].resample("D").sum()
            for m in methods
        }
    )

    # Clear-sky daily max energy from max_calc
    max_daily_energy = max_calc["daily_energy_kwh"]
    max_daily_energy = max_daily_energy.reindex(daily_energy.index, method="nearest")
    daily_energy["max_clearsky_kwh"] = max_daily_energy

    # Monthly average of daily totals (one average "typical" day per month)
    monthly_daily_avg = daily_energy.resample("M").mean()
    monthly_daily_avg.index = monthly_daily_avg.index.to_period("M")

    st.markdown("### Monthly daily average production (kWh/day)")

    monthly_view = monthly_daily_avg[
        ["max_clearsky_kwh", "pvwatts", "combined", "cloud"]
    ].rename(
        columns={
            "max_clearsky_kwh": "Max clear-sky",
            "pvwatts": "PVWatts",
            "combined": "Combined",
            "cloud": "Cloud-adjusted",
        }
    )

    st.dataframe(monthly_view.round(1).rename_axis("Month").astype(float), use_container_width=True)

    # --- MAIN TIME-SERIES PLOTS (unchanged from before) ---
    st.markdown("### Power, energy and performance charts")

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(
        "Enhanced Solar Production Estimation - Multi-Method Comparison",
        fontsize=16,
        fontweight="bold",
    )

    # [keep your time-series plots here as before ...]
    # ... (same as previous full app.py for ax1–ax6)
    # After building fig:
    # st.pyplot(fig)

    # --- SEASONAL JAN–DEC GROUPED BAR CHART ---

    st.markdown("### Seasonal monthly daily averages (Jan–Dec)")

    # Build a clean Jan–Dec index for plotting (strings like '2025-01', etc.)
    monthly_df = monthly_daily_avg.copy()
    # Use month numbers 1–12 to aggregate across years, averaging where needed
    monthly_df["month"] = monthly_df.index.month
    month_group = monthly_df.groupby("month")[
        ["pvwatts", "combined", "cloud"]
    ].mean()

    # Ensure all 12 months are present and in order
    all_months = range(1, 13)
    month_group = month_group.reindex(all_months)

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Create grouped bar chart
    x = np.arange(len(all_months))
    width = 0.25

    fig2, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - width, month_group["pvwatts"], width, label="PVWatts")
    ax.bar(x,         month_group["combined"], width, label="Combined")
    ax.bar(x + width, month_group["cloud"], width, label="Cloud-adjusted")

    ax.set_xticks(x)
    ax.set_xticklabels(month_labels)
    ax.set_ylabel("kWh per typical day")
    ax.set_title("Seasonal monthly daily average production by method")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig2.tight_layout()
    st.pyplot(fig2)

    # --- RECOMMENDATIONS & EXPORT (as before) ---

    avg_powers = {
        m: df_comparison[f"power_{m}"].mean() / 1000 for m in methods
    }
    recommended_method = max(avg_powers.items(), key=lambda x: x[1])[0]

    st.markdown("### Recommendations")
    st.write(f"Recommended method: **{recommended_method.upper()}**")
    st.write(f"Average predicted power: **{avg_powers[recommended_method]:.2f} kW**")

    daily_energy_combined = df_comparison["energy_combined"].resample("D").sum()
    if len(daily_energy_combined) >= 2:
        tomorrow_energy = daily_energy_combined.iloc[1]
    else:
        tomorrow_energy = float("nan")

    st.markdown("### Tomorrow's forecast (from combined method)")
    st.write(f"Expected energy: **{tomorrow_energy:.1f} kWh**")

    results_summary = {
        "system": {
            "total_kwp": TOTAL_KWP,
            "live_peak_kw": float(live_peak_kw),
            "theoretical_max_kw": float(max_calc["total_max_kw"]),
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
        "monthly_daily_avg_kwh": monthly_daily_avg.to_dict(orient="index"),
        "recommendations": {
            "best_method": recommended_method,
            "system_efficiency_percent": float(system_efficiency),
        },
    }

    st.download_button(
        "Download JSON results",
        data=json.dumps(results_summary, indent=2),
        file_name="solar_analysis_results.json",
        mime="application/json",
    )

if __name__ == "__main__":
    run_app()
