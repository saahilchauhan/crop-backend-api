from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # Allow requests from React app

# Load models at startup
print("Loading models...")
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

with open(os.path.join(MODEL_DIR, 'final_optimized_model.pkl'), 'rb') as f:
    model = pickle.load(f)
    
with open(os.path.join(MODEL_DIR, 'feature_info.pkl'), 'rb') as f:
    feature_info = pickle.load(f)

print("✅ Models loaded successfully!")
print(f"Model type: {type(model)}")
print(f"Features needed: {len(feature_info['selected_features'])}")

# Feature engineering function (same as training)
def create_enhanced_features(input_data):
    """Create the same 30+ features used during training"""
    df = pd.DataFrame([input_data])
    
    # NPK ratios
    df['N_P_ratio'] = df['N'] / (df['P'] + 1)
    df['N_K_ratio'] = df['N'] / (df['K'] + 1)
    df['P_K_ratio'] = df['P'] / (df['K'] + 1)
    df['NPK_sum'] = df['N'] + df['P'] + df['K']
    df['NPK_balance'] = df[['N', 'P', 'K']].std(axis=1)
    
    # Temperature features
    df['temp_squared'] = df['temperature'] ** 2
    df['is_cool_season'] = (df['temperature'] < 20).astype(int)
    df['is_warm_season'] = ((df['temperature'] >= 20) & (df['temperature'] <= 30)).astype(int)
    df['is_hot_season'] = (df['temperature'] > 30).astype(int)
    
    # Rainfall features
    df['rainfall_log'] = np.log1p(df['rainfall'])
    df['is_drought_resistant'] = (df['rainfall'] < 50).astype(int)
    df['is_moderate_rain'] = ((df['rainfall'] >= 50) & (df['rainfall'] <= 150)).astype(int)
    df['is_high_rain'] = (df['rainfall'] > 150).astype(int)
    
    # pH features
    df['ph_deviation'] = abs(df['ph'] - 7.0)
    df['is_acidic'] = (df['ph'] < 6.5).astype(int)
    df['is_neutral'] = ((df['ph'] >= 6.5) & (df['ph'] <= 7.5)).astype(int)
    df['is_alkaline'] = (df['ph'] > 7.5).astype(int)
    
    # Humidity features
    df['humidity_squared'] = df['humidity'] ** 2
    df['is_dry'] = (df['humidity'] < 60).astype(int)
    df['is_humid'] = (df['humidity'] > 80).astype(int)
    
    # Interaction features
    df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100
    df['temp_rainfall_interaction'] = df['temperature'] * df['rainfall'] / 100
    df['ph_npk_interaction'] = df['ph'] * df['NPK_sum'] / 100
    
    # Climate indices
    df['tropical_index'] = (
        df['temperature'] * 0.4 + 
        df['humidity'] * 0.3 + 
        df['rainfall'] * 0.3
    ) / 100
    
    df['temperate_index'] = (
        (30 - abs(df['temperature'] - 20)) * 0.4 + 
        df['humidity'] * 0.3 + 
        (200 - abs(df['rainfall'] - 100)) * 0.3
    ) / 100
    
    return df

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'message': 'Crop Recommendation API is running!',
        'model_loaded': model is not None,
        'features_count': len(feature_info['selected_features'])
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make crop predictions"""
    try:
        # Get input data
        data = request.json
        print(f"Received data: {data}")
        
        # Validate required fields
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Create enhanced features
        df_enhanced = create_enhanced_features(data)
        
        # Select only the features used during training
        X_input = df_enhanced[feature_info['selected_features']]
        
        print(f"Input shape: {X_input.shape}")
        
        # Make prediction
        prediction = model.predict(X_input)[0]
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_input)[0]
            
            # Get top 5 predictions
            top_5_idx = np.argsort(probabilities)[-5:][::-1]
            top_5_crops = [model.classes_[i] for i in top_5_idx]
            top_5_probs = [float(probabilities[i]) for i in top_5_idx]
            
            print(f"Top prediction: {prediction} ({top_5_probs[0]*100:.1f}%)")
            
            return jsonify({
                'success': True,
                'top_crop': prediction,
                'top_5_crops': top_5_crops,
                'top_5_probabilities': top_5_probs,
                'confidence': top_5_probs[0]
            })
        else:
            return jsonify({
                'success': True,
                'prediction': prediction,
                'confidence': 1.0
            })
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    return jsonify({
        'model_type': str(type(model).__name__),
        'model_accuracy': float(feature_info.get('accuracy', 0)),
        'model_name': feature_info.get('model_name', 'Unknown'),
        'features_count': len(feature_info['selected_features']),
        'crops_supported': len(model.classes_) if hasattr(model, 'classes_') else 0,
        'crops_list': list(model.classes_) if hasattr(model, 'classes_') else []
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌾 CROP RECOMMENDATION API SERVER")
    print("="*60)
    print(f"Model: {feature_info.get('model_name', 'Unknown')}")
    print(f"Accuracy: {feature_info.get('accuracy', 'N/A')}")
    print(f"Features: {len(feature_info['selected_features'])}")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)