from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from safe_route_planner import SafeRoutePlanner
from models import db, RoadFeedback
import os
from werkzeug.utils import secure_filename
from datetime import datetime
from PIL import Image, ImageEnhance, ImageDraw
import numpy as np
from scipy import ndimage
import base64
from io import BytesIO
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app and SafeRoutePlanner instance
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///routex.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_potholes(image_path):
    try:
        logger.debug(f"Reading image from {image_path}")
        # Open the image
        original_image = Image.open(image_path)
        
        # Convert to grayscale
        gray_image = original_image.convert('L')
        logger.debug("Converted to grayscale")
        
        # Enhance contrast
        contrast_enhancer = ImageEnhance.Contrast(gray_image)
        enhanced_image = contrast_enhancer.enhance(2.0)
        
        # Enhance sharpness
        sharpness_enhancer = ImageEnhance.Sharpness(enhanced_image)
        enhanced_image = sharpness_enhancer.enhance(1.5)
        logger.debug("Enhanced image")
        
        # Convert to numpy array for processing
        img_array = np.array(enhanced_image)
        
        # Apply adaptive thresholding
        window_size = 25
        local_mean = ndimage.uniform_filter(img_array, size=window_size)
        threshold = local_mean - 20  # Adjust this offset for sensitivity
        binary = (img_array < threshold).astype(np.uint8) * 255
        logger.debug("Applied adaptive threshold")
        
        # Apply morphological operations to clean up the image
        kernel_size = 3
        binary = ndimage.binary_opening(binary, structure=np.ones((kernel_size, kernel_size)))
        binary = ndimage.binary_closing(binary, structure=np.ones((kernel_size, kernel_size)))
        logger.debug("Applied morphological operations")
        
        # Find connected components (potential potholes)
        labeled_array, num_features = ndimage.label(binary)
        logger.debug(f"Found {num_features} initial features")
        
        # Calculate properties of detected regions
        pothole_regions = []
        min_size = 200  # minimum size in pixels
        max_size = 20000  # maximum size in pixels
        
        for i in range(1, num_features + 1):
            region = (labeled_array == i)
            area = np.sum(region)
            
            if min_size < area < max_size:
                # Calculate region properties
                coords = np.where(region)
                bbox = {
                    'min_y': int(np.min(coords[0])),
                    'max_y': int(np.max(coords[0])),
                    'min_x': int(np.min(coords[1])),
                    'max_x': int(np.max(coords[1]))
                }
                
                # Calculate aspect ratio
                width = bbox['max_x'] - bbox['min_x']
                height = bbox['max_y'] - bbox['min_y']
                aspect_ratio = width / height if height != 0 else 0
                
                # Only include regions with reasonable aspect ratios (not too elongated)
                if 0.5 < aspect_ratio < 2.0:
                    pothole_regions.append(bbox)
        
        logger.debug(f"Found {len(pothole_regions)} potential potholes")
        
        # Draw boxes around detected potholes
        draw = ImageDraw.Draw(original_image)
        for bbox in pothole_regions:
            # Draw red rectangle
            draw.rectangle(
                [
                    (bbox['min_x'], bbox['min_y']),
                    (bbox['max_x'], bbox['max_y'])
                ],
                outline='red',
                width=3
            )
        
        # Convert the image to base64 for sending to frontend
        buffered = BytesIO()
        original_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return len(pothole_regions), img_str
        
    except Exception as e:
        logger.error(f"Error in detect_potholes: {str(e)}")
        raise

# Initialize extensions
db.init_app(app)
planner = SafeRoutePlanner()

# Create database tables
with app.app_context():
    db.create_all()

# Ensure upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Ensure index.html is served at the root
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Redirect /index to root for consistency
@app.route('/index')
def index_redirect():
    return redirect(url_for('index'))

@app.route('/getroute')
def get_route():
    return render_template('getroute.html')

@app.route('/safety-check')
def safety_check():
    return render_template('safety-check.html')

@app.route('/navigate')
def navigate():
    return render_template('navigation.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    try:
        # Get form data
        location = request.form.get('location')
        latitude = float(request.form.get('latitude'))
        longitude = float(request.form.get('longitude'))
        condition = request.form.get('condition')
        severity = request.form.get('severity')
        description = request.form.get('description')
        
        # Handle image upload
        if 'image' not in request.files:
            return jsonify(success=False, message="No image file provided")
            
        image = request.files['image']
        if image.filename == '':
            return jsonify(success=False, message="No image selected")
            
        if image:
            filename = secure_filename(image.filename)
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_')
            unique_filename = timestamp + filename
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            image.save(image_path)
            
            # Create feedback record
            feedback = RoadFeedback(
                location=location,
                latitude=latitude,
                longitude=longitude,
                condition=condition,
                severity=severity,
                description=description,
                image_path=unique_filename
            )
            
            db.session.add(feedback)
            db.session.commit()
            
            return jsonify(success=True, message="Feedback submitted successfully")
            
    except Exception as e:
        return jsonify(success=False, message=str(e))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/pothole-detection')
def pothole_detection():
    return render_template('pothole-detection.html')

@app.route('/detect-potholes', methods=['POST'])
def detect_potholes_route():
    try:
        logger.debug("Received pothole detection request")
        if 'image' not in request.files:
            logger.warning("No image file in request")
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            logger.warning("Empty filename")
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.debug(f"Saving file to {filepath}")
            file.save(filepath)
            
            try:
                # Perform pothole detection
                logger.debug("Starting pothole detection")
                pothole_count, processed_image = detect_potholes(filepath)
                logger.info(f"Detection complete. Found {pothole_count} potholes")
                
                # Clean up the uploaded file
                os.remove(filepath)
                logger.debug("Cleaned up uploaded file")
                
                return jsonify({
                    'success': True,
                    'potholes_detected': pothole_count > 0,
                    'pothole_count': pothole_count,
                    'processed_image': processed_image
                })
                
            except Exception as e:
                logger.error(f"Error during detection: {str(e)}")
                # Clean up the uploaded file in case of error
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({
                    'success': False,
                    'error': f"Detection error: {str(e)}"
                }), 500
        
        logger.warning("Invalid file type")
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"Server error: {str(e)}"
        }), 500

@app.route('/analyze_route', methods=['POST'])
def analyze_route():
    try:
        data = request.get_json()
        source = data.get('source')
        destination = data.get('destination')
        
        logger.debug(f"Analyzing route from {source} to {destination}")
        
        if not source or not destination:
            logger.error("Missing source or destination")
            return jsonify({'error': 'Source and destination are required'}), 400
            
        result = planner.analyze_route(source, destination)
        logger.debug(f"Route analysis result: {result}")
        
        if not result:
            logger.error("Route analysis returned None")
            return jsonify({'error': 'Could not analyze route'}), 400
            
        # Get LIME explanation for the prediction
        features = [
            result['source_weather']['temperature'],
            result['source_weather']['humidity'],
            result['source_weather']['wind_speed'],
            result['distance'],
            result['duration'],
            result['source_weather'].get('visibility', 1.0),
            result['source_weather']['pressure'],
            float(datetime.now().hour),
            float(datetime.now().weekday() >= 5)
        ]
        explanation = planner.explain_prediction(np.array([features]))
        
        response = {
            'source_coords': result['source_coords'],
            'dest_coords': result['dest_coords'],
            'route_coords': result['route_coords'],
            'distance': result['distance'],
            'duration': int(result['duration']),  # Format duration as integer
            'safety_score': result['safety_score'],
            'source_weather': result['source_weather'],
            'dest_weather': result['dest_weather'],
            'weather_advice': result['weather_advice'],
            'feature_analysis': {
                'explanation_plot': explanation.get('explanation_plot', ''),
                'contributions': explanation.get('contributions', []),
                'feature_explanations': explanation.get('feature_explanations', []),
                'overall_explanation': explanation.get('overall_explanation', '')
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in analyze_route: {str(e)}")
        logger.error(f"Stack trace:", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/get_weather', methods=['GET'])
def get_weather():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    if not lat or not lon:
        return jsonify(success=False, message="Latitude and longitude are required.")
    
    try:
        weather_data = planner.get_weather(float(lat), float(lon))
        return jsonify(success=True, weather=weather_data)
    except Exception as e:
        return jsonify(success=False, message=f"Error fetching weather data: {str(e)}")

@app.route('/get_weather_update', methods=['POST'])
def get_weather_update():
    try:
        data = request.get_json()
        lat = data.get('lat')
        lon = data.get('lon')
        location_type = data.get('type', 'source')  # 'source' or 'destination'
        
        if not lat or not lon:
            return jsonify({'error': 'Latitude and longitude are required'}), 400
            
        logger.debug(f"Getting weather update for {location_type} location: {lat}, {lon}")
        weather_data = planner.get_weather(lat, lon)
        
        if not weather_data:
            logger.error(f"Could not fetch weather data for {location_type} location")
            return jsonify({'error': f'Could not fetch weather data for {location_type} location'}), 400
            
        # Add weather advice
        weather_data['weather_advice'] = planner.get_weather_advice(weather_data)
        logger.debug(f"Weather data for {location_type}: {weather_data}")
        
        return jsonify(weather_data)
        
    except Exception as e:
        logger.error(f"Error getting weather update: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/view-feedback')
def view_feedback():
    feedbacks = RoadFeedback.query.all()
    return render_template('view_feedback.html', feedbacks=feedbacks)

@app.route('/api/navigation/update', methods=['POST'])
def update_navigation():
    try:
        data = request.get_json()
        current_lat = data.get('latitude')
        current_lng = data.get('longitude')
        destination_lat = data.get('dest_latitude')
        destination_lng = data.get('dest_longitude')

        # Get nearby hazards and road conditions
        nearby_hazards = planner.get_nearby_hazards(current_lat, current_lng)
        
        # Get current weather
        weather_info = planner.get_weather_at_location(current_lat, current_lng)
        
        # Calculate remaining distance and ETA
        remaining_info = planner.calculate_remaining_info(
            current_lat, current_lng,
            destination_lat, destination_lng
        )
        
        return jsonify({
            'success': True,
            'hazards': nearby_hazards,
            'weather': weather_info,
            'remaining_distance': remaining_info['distance'],
            'eta': remaining_info['eta'],
            'next_instruction': remaining_info['next_instruction']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/navigation/end', methods=['POST'])
def end_navigation():
    try:
        # Save navigation statistics or handle cleanup
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_route_update', methods=['POST'])
def get_route_update():
    try:
        data = request.get_json()
        source_coords = data.get('source_coords')
        dest_coords = data.get('dest_coords')
        distance = data.get('distance')
        duration = data.get('duration')
        
        if not all([source_coords, dest_coords, distance, duration]):
            return jsonify({'error': 'Missing required route parameters'}), 400
            
        logger.debug(f"Getting route update for {source_coords} to {dest_coords}")
        
        # Get current weather for both locations
        source_weather = planner.get_weather(source_coords[0], source_coords[1])
        dest_weather = planner.get_weather(dest_coords[0], dest_coords[1])
        
        if not source_weather or not dest_weather:
            return jsonify({'error': 'Could not fetch weather data'}), 400
            
        # Analyze route safety with current conditions
        features = [
            source_weather['temperature'],
            source_weather['humidity'],
            source_weather['wind_speed'],
            distance,
            duration,
            source_weather.get('visibility', 1.0),
            source_weather['pressure'],
            float(datetime.now().hour),
            float(datetime.now().weekday() >= 5)
        ]
        
        explanation = planner.explain_prediction(np.array([features]))
        
        response = {
            'source_weather': source_weather,
            'dest_weather': dest_weather,
            'safety_score': explanation['safety_score'],
            'feature_analysis': {
                'overall_explanation': explanation['overall_explanation'],
                'feature_explanations': explanation['feature_explanations']
            },
            'weather_advice': planner.get_weather_advice(source_weather)
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error getting route update: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Clear any existing sessions
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.config['PERMANENT_SESSION_LIFETIME'] = 0
    app.debug = True
    app.run(host='127.0.0.1', port=5000)
