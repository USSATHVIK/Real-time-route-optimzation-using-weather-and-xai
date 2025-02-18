import requests
import pandas as pd
import numpy as np
from typing import Optional, Dict
from sklearn.ensemble import GradientBoostingRegressor
import folium
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
import folium
import logging
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from sklearn.preprocessing import StandardScaler
import lime
from lime import lime_tabular

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# API Keys
OPENWEATHER_API_KEY = 'f5507108c9cfe91b420ff067b37ef084'

class SafeRoutePlanner:
    def __init__(self):
        self.ors_api_key = '5b3ce3597851110001cf6248dafe6738bb1647afaef954b10918a97a'
        self.ors_headers = {
            'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
            'Authorization': f'{self.ors_api_key}',
            'Content-Type': 'application/json; charset=utf-8'
        }
        # Using GradientBoostingRegressor for better accuracy and interpretability
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        self.feature_names = [
            'temperature', 'humidity', 'wind_speed', 'distance', 'duration',
            'visibility', 'pressure', 'time_of_day', 'is_weekend'
        ]
        
        # Initialize scaler for feature normalization
        self.scaler = StandardScaler()
        
        # Initialize LIME explainer
        self.explainer = None
        
        # Train the model with some initial safety data
        self._train_initial_model()
        
        self.weather_risk_descriptions = {
            'Clear': 'Optimal driving conditions with good visibility',
            'Clouds': 'Good driving conditions with slightly reduced visibility',
            'Mist': 'Reduced visibility requiring careful driving',
            'Rain': 'Slippery roads and reduced visibility, drive with caution',
            'Snow': 'Hazardous conditions, reduced traction and visibility',
            'Thunderstorm': 'Dangerous conditions with very poor visibility and risk of hydroplaning'
        }

    def _train_initial_model(self):
        """Train the model with initial safety data and prepare for explanations."""
        # Sample training data
        X_train = np.array([
            [25, 60, 10, 100, 120, 1.0, 1013, 12, 0],  # Good conditions
            [20, 70, 15, 150, 180, 0.8, 1010, 14, 0],  # Moderate conditions
            [30, 80, 20, 200, 240, 0.6, 1008, 16, 1],  # Challenging conditions
            [15, 90, 25, 250, 300, 0.4, 1005, 18, 1],  # Difficult conditions
            [10, 95, 30, 300, 360, 0.2, 1000, 20, 0],  # Severe conditions
        ])
        
        # Safety scores (0-100, where 100 is safest)
        y_train = np.array([90, 80, 70, 60, 50])
        
        # Fit scaler and transform training data
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Initialize LIME explainer
        self.explainer = lime_tabular.LimeTabularExplainer(
            X_train_scaled,
            feature_names=self.feature_names,
            class_names=['safety_score'],
            mode='regression'
        )

    def get_coordinates(self, place_name: str) -> Optional[tuple]:
        """Get coordinates for a place name using OpenRouteService Geocoding API."""
        url = "https://api.openrouteservice.org/geocode/search"
        params = {
            'text': place_name,
            'size': 1
        }
        
        try:
            response = requests.get(url, headers=self.ors_headers, params=params)
            print(f"Geocoding URL: {response.url}")  # Debug log
            print(f"Geocoding status code: {response.status_code}")  # Debug log
            
            if response.status_code != 200:
                print(f"Geocoding error response: {response.text}")
                return None
                
            data = response.json()
            print(f"Geocoding response for {place_name}:", data)  # Debug log
            
            if not data.get('features'):
                print(f"No coordinates found for {place_name}")
                return None
                
            coordinates = data['features'][0]['geometry']['coordinates']
            return (coordinates[1], coordinates[0])  # Return (lat, lon)
            
        except Exception as e:
            print(f"Error getting coordinates for {place_name}: {str(e)}")
            return None

    def get_weather(self, lat: float, lon: float) -> Optional[Dict]:
        """Get current weather data for given coordinates using OpenWeatherMap API."""
        try:
            # OpenWeatherMap API key
            api_key = 'f5507108c9cfe91b420ff067b37ef084'
            url = f'http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric'
            
            logger.debug(f"Fetching weather data from: {url}")  
            response = requests.get(url)
            logger.debug(f"Weather API status code: {response.status_code}")  
            
            if response.status_code != 200:
                logger.error(f"Weather API error response: {response.text}")
                return None
                
            data = response.json()
            logger.debug(f"Weather API response: {data}")  
            
            if not data or 'main' not in data or 'weather' not in data or not data['weather']:
                logger.error("Invalid or incomplete weather data")
                return None
                
            # Extract weather condition from the first weather item
            weather_condition = data['weather'][0].get('main', 'Unknown')
            logger.debug(f"Weather condition: {weather_condition}")  
            
            weather_data = {
                'temperature': round(float(data['main'].get('temp', 0)), 1),
                'humidity': int(data['main'].get('humidity', 0)),
                'wind_speed': round(float(data['wind'].get('speed', 0)), 1),
                'weather_condition': weather_condition,
                'visibility': round(float(data.get('visibility', 10000)) / 10000.0, 2),  # Convert to km
                'pressure': int(data['main'].get('pressure', 1013))
            }
            
            logger.debug(f"Processed weather data: {weather_data}")  
            return weather_data
            
        except KeyError as e:
            logger.error(f"Missing key in weather data: {str(e)}")
            return None
        except ValueError as e:
            logger.error(f"Invalid numeric value in weather data: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error getting weather data: {str(e)}")
            return None

    def calculate_safety_score(self, weather_data: Dict, distance_km: float, duration_min: float) -> tuple:
        """Calculate safety score based on weather and route conditions."""
        try:
            current_hour = int(datetime.now().strftime('%H'))
            is_weekend = int(datetime.now().weekday() >= 5)
            
            # Prepare features for prediction
            features = np.array([[
                weather_data.get('temperature', 20),
                weather_data.get('humidity', 70),
                weather_data.get('wind_speed', 10),
                distance_km,
                duration_min,
                weather_data.get('visibility', 1.0),
                weather_data.get('pressure', 1013),
                current_hour,
                is_weekend
            ]])
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get prediction and feature importance
            safety_score = max(0, min(100, self.model.predict(features_scaled)[0]))
            feature_importance = self.model.feature_importances_
            
            return safety_score, feature_importance
            
        except Exception as e:
            logger.error(f"Error calculating safety score: {e}")
            return 75.0, np.zeros(len(self.feature_names))  # Default safety score and zero importance

    def get_weather_advice(self, weather: Dict) -> str:
        condition = weather['weather_condition']
        temp = weather['temperature']
        wind = weather['wind_speed']
        
        advice = [self.weather_risk_descriptions.get(condition, "Unknown weather condition")]
        
        if temp < 0:
            advice.append("Risk of ice formation")
        elif temp > 35:
            advice.append("Risk of vehicle overheating")
        
        if wind > 15:
            advice.append("Strong winds may affect vehicle stability")
        
        return " • ".join(advice)

    def explain_prediction(self, features) -> Dict:
        """
        Generate a detailed explanation of the safety score prediction using LIME.
        
        Args:
            features: Array of input features
            
        Returns:
            Dict containing explanation components
        """
        try:
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get prediction
            safety_score = float(self.model.predict(features_scaled)[0])
            
            # Generate LIME explanation
            exp = self.explainer.explain_instance(
                features_scaled[0], 
                self.model.predict,
                num_features=len(self.feature_names)
            )
            
            # Create explanation plot
            plt.figure(figsize=(12, 6))
            plt.title('Feature Importance Analysis')
            exp.as_pyplot_figure()
            
            # Customize plot appearance
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save explanation plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            buffer.seek(0)
            explanation_plot = base64.b64encode(buffer.getvalue()).decode()
            
            # Calculate feature contributions
            contributions = []
            feature_explanations = []
            
            for name, value in zip(self.feature_names, features[0]):
                importance = dict(exp.as_list()).get(name, 0)
                contributions.append({
                    'feature': name,
                    'value': float(value),
                    'importance': float(importance),
                    'abs_impact': abs(importance)
                })
                
                # Generate natural language explanation
                explanation = self._generate_feature_explanation(name, value, importance)
                feature_explanations.append(explanation)
            
            # Sort contributions by absolute impact
            contributions.sort(key=lambda x: x['abs_impact'], reverse=True)
            
            # Generate overall explanation
            overall_explanation = self._generate_overall_explanation(safety_score, contributions[:3])
            
            return {
                'safety_score': safety_score,
                'explanation_plot': explanation_plot,
                'contributions': contributions,
                'feature_explanations': feature_explanations,
                'overall_explanation': overall_explanation
            }
            
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            return {
                'safety_score': 75.0,  # Default safety score
                'explanation_plot': '',
                'contributions': [],
                'feature_explanations': ['Could not generate detailed analysis'],
                'overall_explanation': 'Analysis currently unavailable'
            }

    def _generate_feature_explanation(self, feature, value, importance) -> str:
        """Generate natural language explanation for a feature's contribution."""
        impact = abs(importance)
        direction = "increases" if importance > 0 else "decreases"
        
        if impact < 0.1:
            strength = "slightly"
        elif impact < 0.3:
            strength = "moderately"
        else:
            strength = "significantly"
        
        explanations = {
            'temperature': f"The temperature of {value:.1f}°C {strength} {direction} route safety",
            'humidity': f"The humidity level of {value:.1f}% {strength} {direction} safety conditions",
            'wind_speed': f"Wind speeds of {value:.1f} m/s {strength} {direction} driving safety",
            'distance': f"The route distance of {value:.1f} km {strength} {direction} overall safety",
            'duration': f"The journey duration of {int(value)} minutes {strength} {direction} safety",
            'visibility': f"Visibility conditions of {value:.1f} km {strength} {direction} safety",
            'pressure': f"Atmospheric pressure of {value:.1f} hPa {strength} {direction} conditions",
            'time_of_day': f"The current time {strength} {direction} route safety",
            'is_weekend': f"{'Weekend' if value else 'Weekday'} travel {strength} {direction} safety"
        }
        
        return explanations.get(feature, f"The {feature} value of {value} {strength} {direction} safety")

    def _generate_overall_explanation(self, safety_score, top_contributions) -> str:
        """
        Generate a natural language explanation of the overall safety score.
        
        Args:
            safety_score: The predicted safety score
            top_contributions: List of top contributing features
            
        Returns:
            A string containing the overall explanation
        """
        try:
            # Determine safety level
            if safety_score >= 80:
                safety_level = "very safe"
            elif safety_score >= 60:
                safety_level = "moderately safe"
            else:
                safety_level = "potentially risky"
            
            # Create base explanation
            explanation = f"This route is considered {safety_level} with a safety score of {safety_score:.1f}. "
            
            # Add top contributing factors
            if top_contributions:
                explanation += "The main factors affecting safety are: "
                factor_explanations = []
                
                for contrib in top_contributions:
                    feature = contrib['feature']
                    value = contrib['value']
                    impact = contrib['importance']
                    
                    # Format the feature name for display
                    feature_display = feature.replace('_', ' ').title()
                    
                    # Generate specific explanations based on feature type
                    if feature == 'temperature':
                        factor = f"temperature ({value:.1f}°C)"
                        effect = "higher" if impact > 0 else "lower"
                        factor_explanations.append(f"{factor} leading to {effect} safety")
                    elif feature == 'wind_speed':
                        factor = f"wind speed ({value:.1f} m/s)"
                        effect = "reduced" if impact < 0 else "increased"
                        factor_explanations.append(f"{factor} causing {effect} safety")
                    elif feature == 'humidity':
                        factor = f"humidity ({value:.0f}%)"
                        effect = "decreased" if impact < 0 else "increased"
                        factor_explanations.append(f"{factor} resulting in {effect} safety")
                    elif feature == 'distance':
                        factor = f"route distance ({value:.1f} km)"
                        effect = "longer" if impact < 0 else "shorter"
                        factor_explanations.append(f"a {effect} {factor}")
                    elif feature == 'duration':
                        factor = f"travel time ({value:.0f} min)"
                        effect = "extended" if impact < 0 else "reasonable"
                        factor_explanations.append(f"an {effect} {factor}")
                    elif feature == 'visibility':
                        factor = f"visibility ({value:.1f} km)"
                        effect = "poor" if impact < 0 else "good"
                        factor_explanations.append(f"{effect} {factor}")
                    else:
                        # Generic explanation for other features
                        effect = "positive" if impact > 0 else "negative"
                        factor_explanations.append(f"{feature_display} having a {effect} impact")
                
                # Combine factor explanations
                if len(factor_explanations) > 1:
                    explanation += f"{', '.join(factor_explanations[:-1])}, and {factor_explanations[-1]}"
                else:
                    explanation += factor_explanations[0]
                
                explanation += "."
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating overall explanation: {str(e)}")
            return "Unable to generate detailed explanation at this time."

    def analyze_route_safety(self, start_coords, end_coords, weather_data):
        """Analyze route safety and return safety score with feature importance."""
        # Extract features
        features = [
            float(weather_data['temperature']),  # temperature
            float(weather_data['humidity']),  # humidity
            float(weather_data['wind_speed']),  # wind_speed
            float(weather_data.get('distance', 100)),  # distance (default if not available)
            float(weather_data.get('duration', 120)),  # duration (default if not available)
            float(weather_data.get('visibility', 1.0)),  # visibility
            float(weather_data['pressure']),  # pressure
            float(datetime.now().hour),  # time_of_day
            float(datetime.now().weekday() >= 5)  # is_weekend
        ]
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Get safety score
        safety_score = float(self.model.predict(features_scaled)[0])
        
        # Get feature importance
        feature_importance = self.model.feature_importances_
        
        return {
            'safety_score': safety_score,
            'feature_importance': dict(zip(self.feature_names, feature_importance))
        }

    def get_safety_explanation(self, features):
        """
        Generate LIME-based explanation for the safety score prediction.
        
        Args:
            features: Array of feature values for prediction
            
        Returns:
            tuple: (explanation_plot_base64, feature_importance_dict)
        """
        # Reshape features for single prediction
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features_array)
        
        # Generate LIME explanation
        exp = self.explainer.explain_instance(
            features_scaled[0], 
            self.model.predict,
            num_features=len(self.feature_names)
        )
        
        # Get feature importance from LIME
        feature_importance = dict(exp.as_list())
        
        # Create explanation plot
        plt.figure(figsize=(10, 6))
        exp.as_pyplot_figure()
        
        # Save explanation plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        explanation_plot = base64.b64encode(buffer.getvalue()).decode()
        
        # Return explanation plot and feature importance
        return explanation_plot, feature_importance

    def analyze_route(self, source: str, destination: str) -> Dict:
        """Analyze the safety of a route between two locations."""
        try:
            logger.debug(f"Getting coordinates for source: {source}")
            source_coords = self.get_coordinates(source)
            if not source_coords:
                logger.error(f"Could not get coordinates for source location: {source}. Please check if the location name is valid.")
                return {'error': f'Could not find coordinates for source location: {source}'}
                
            logger.debug(f"Getting coordinates for destination: {destination}")
            dest_coords = self.get_coordinates(destination)
            if not dest_coords:
                logger.error(f"Could not get coordinates for destination location: {destination}. Please check if the location name is valid.")
                return {'error': f'Could not find coordinates for destination location: {destination}'}
                
            logger.debug(f"Getting weather for source coordinates: {source_coords}")
            source_weather = self.get_weather(*source_coords)
            if not source_weather:
                logger.error(f"Could not get weather data for source location. OpenWeather API may be unavailable.")
                return {'error': 'Weather service unavailable for source location'}
                
            logger.debug(f"Getting weather for destination coordinates: {dest_coords}")
            dest_weather = self.get_weather(*dest_coords)
            if not dest_weather:
                logger.error(f"Could not get weather data for destination location. OpenWeather API may be unavailable.")
                return {'error': 'Weather service unavailable for destination location'}
                
            logger.debug("Getting route data")
            route_url = "https://api.openrouteservice.org/v2/directions/driving-car/geojson"
            route_body = {
                "coordinates": [[source_coords[1], source_coords[0]], [dest_coords[1], dest_coords[0]]]
            }
            
            try:
                logger.debug(f"Requesting route with coordinates: {route_body['coordinates']}")
                route_response = requests.post(route_url, headers=self.ors_headers, json=route_body)
                logger.debug(f"Route API response status: {route_response.status_code}")
                logger.debug(f"Route API response: {route_response.text[:500]}")  # Log first 500 chars to avoid huge logs
                
                if route_response.status_code != 200:
                    error_msg = f'Route calculation failed: {route_response.text}'
                    logger.error(error_msg)
                    return {'error': error_msg}
                    
                route_data = route_response.json()
                
                if not route_data.get('features'):
                    logger.error("No route found between the specified locations")
                    return {'error': 'No route found between the specified locations'}
                    
                # Extract route details
                route = route_data['features'][0]
                distance = route['properties']['segments'][0]['distance'] / 1000  # Convert meters to kilometers
                duration = route['properties']['segments'][0]['duration'] / 60  # Convert seconds to minutes
                
                # Extract coordinates from GeoJSON
                coordinates = route['geometry']['coordinates']
                route_coords = [[coord[1], coord[0]] for coord in coordinates]
                
                logger.debug("Analyzing route safety")
                route_safety = self.analyze_route_safety(source_coords, dest_coords, source_weather)
                
                response = {
                    'source_coords': source_coords,
                    'dest_coords': dest_coords,
                    'route_coords': route_coords,
                    'distance': distance,
                    'duration': int(duration),  # Format duration as integer
                    'safety_score': route_safety['safety_score'],
                    'source_weather': source_weather,
                    'dest_weather': dest_weather,
                    'route_data': route_data,
                    'feature_importance': route_safety['feature_importance'],
                    'weather_advice': self.get_weather_advice(source_weather)
                }
                
                logger.debug(f"Route analysis complete: {response}")
                return response
                
            except Exception as e:
                logger.error(f"Error in analyze_route: {str(e)}")
                logger.error("Stack trace:", exc_info=True)
                return None
            
        except Exception as e:
            logger.error(f"Error in analyze_route: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            return None

def main():
    print(" Safe Route Analyzer with XAI ")
    planner = SafeRoutePlanner()
    
    while True:
        try:
            source = input("\nEnter start location (with state/country): ")
            destination = input("Enter destination (with state/country): ")
            
            result = planner.analyze_route(source, destination)
            if result and 'error' not in result:
                print(f"\n Route Analysis:")
                print(f"Distance: {result['distance']:.1f} km")
                print(f"Duration: {result['duration']} minutes")
                
                print(f"\n Source Location ({source}):")
                print(f"Weather: {result['source_weather']['weather_condition']}")
                print(f"Temperature: {result['source_weather']['temperature']}°C")
                print(f"Wind Speed: {result['source_weather']['wind_speed']} m/s")
                print(f"Safety Score: {result['safety_score']:.1f}%")
                print(f"Safety Advisory: {result['weather_advice']}")
                
                print(f"\n Destination ({destination}):")
                print(f"Weather: {result['dest_weather']['weather_condition']}")
                print(f"Temperature: {result['dest_weather']['temperature']}°C")
                print(f"Wind Speed: {result['dest_weather']['wind_speed']} m/s")
                
                print("\n Feature Importance Analysis:")
                for feature, importance in result['feature_importance'].items():
                    print(f"{feature}: {importance:.3f}")
        
        except Exception as e:
            print(f"Error analyzing route: {e}")
        
        if input("\nAnalyze another route? (y/n): ").lower() != 'y':
            break

if __name__ == "__main__":
    main()