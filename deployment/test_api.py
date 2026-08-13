import io
import time
import requests
import numpy as np

# Configuration targeting the local development FastAPI server default port
BASE_URL = "http://127.0.0.1:8000"

def run_integration_tests():
    print("=" * 60)
    print("STARTING LIVE INFRASTRUCTURE INTEGRATION TESTING SUITE")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # TEST 1: Retrieve Model Metadata Endpoint (GET /meta)
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Verifying /meta metadata serialization retrieval...")
    try:
        meta_response = requests.get(f"{BASE_URL}/meta")
        print(f"Status Code received: {meta_response.status_code}")
        
        if meta_response.status_code == 200:
            meta_data = meta_response.json()
            print(f" ✅ Successfully communicated with active architecture: {meta_data['model_name']}")
            print(f" Target Sequence Length required: {meta_data['sequence_length']}")
            print(f" Target Spatial Resolution expected: {meta_data['input_resolution']}")
            
            # Cache metadata metrics dynamically for subsequent array test synthesis
            seq_len = meta_data['sequence_length']
            h, w = meta_data['input_resolution']
        else:
            print(f"❌ Failed Metadata Query: {meta_response.text}")
            return
    except requests.exceptions.ConnectionError:
        print("❌ CRITICAL: Could not connect to FastAPI server. Is 'make api-dev' running?")
        return

    # -------------------------------------------------------------------------
    # TEST 2: Successful Model Inference (POST /predict with perfect payload)
    # -------------------------------------------------------------------------
    print(f"\n[TEST 2] Generating virtual radar input sequence [{seq_len}, {h}, {w}]...")
    
    # Simulate a raw radar matrix with missing frames (-1 values)
    synthetic_radar = np.random.uniform(0.0, 30.0, size=(seq_len, h, w)).astype(np.float32)
    synthetic_radar[0, 10:20, 10:20] = -1.0  # Inject missing radar token to trigger mask logic
    
    # Serialize the numpy array straight into an in-memory binary file stream
    buffer = io.BytesIO()
    np.save(buffer, synthetic_radar)
    buffer.seek(0)  # Rewind to starting position for file transfer read triggers

    print("Streaming payload tensor array to POST /predict...")
    start_time = time.time()
    predict_response = requests.post(
        f"{BASE_URL}/predict",
        files={"file": ("synthetic_radar.npy", buffer, "application/octet-stream")}
    )
    latency = (time.time() - start_time) * 1000
    
    print(f"Status Code received: {predict_response.status_code} (Latency: {latency:.2f}ms)")
    if predict_response.status_code == 200:
        result = predict_response.json()
        print(f" ✅ Nowcast Matrix Response: Success={result['success']}")
        print(f" Returned prediction array layout geometry: {result['prediction_shape']}")
    else:
        print(f"❌ Failed Perfect Prediction Payload Test: {predict_response.text}")

    # -------------------------------------------------------------------------
    # TEST 3: Client Validation Error Protection (POST /predict with bad shape)
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Testing system dimensions crash protection (Incorrect Sequence Length)...")
    
    # Intentionally corrupt sequence dimension (send 3 frames instead of required length)
    bad_shape_radar = np.random.uniform(0.0, 30.0, size=(3, h, w)).astype(np.float32)
    
    bad_buffer = io.BytesIO()
    np.save(bad_buffer, bad_shape_radar)
    bad_buffer.seek(0)

    bad_predict_response = requests.post(
        f"{BASE_URL}/predict",
        files={"file": ("corrupted_radar.npy", bad_buffer, "application/octet-stream")}
    )
    print(f"Status Code received: {bad_predict_response.status_code}")
    if bad_predict_response.status_code == 422:
        print(" ✅ Successfully triggered 422 Unprocessable Entity protection!")
        print(f" Server feedback detail message: {bad_predict_response.json()['detail']}")
    else:
        print(f"❌ System failure protection missed. Expected 422, got: {bad_predict_response.status_code}")

    # -------------------------------------------------------------------------
    # TEST 4: Client Request Extension Type Verification (POST /predict with bad file type)
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Testing file type security boundaries (Uploading text payload instead of .npy)...")
    
    fake_file_buffer = io.BytesIO(b"Fake plain text data string payload structure tracker")
    
    bad_type_response = requests.post(
        f"{BASE_URL}/predict",
        files={"file": ("text_report.txt", fake_file_buffer, "text/plain")}
    )
    print(f"Status Code received: {bad_type_response.status_code}")
    if bad_type_response.status_code == 400:
        print(" ✅ Successfully triggered 400 Bad Request extension blockade protection!")
        print(f" Server feedback detail message: {bad_type_response.json()['detail']}")
    else:
        print(f"❌ Extension file tracking filter missed. Expected 400, got: {bad_type_response.status_code}")

    print("\n" + "=" * 60)
    print("INTEGRATION TESTING COMPLETION CYCLE TERMINATED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    run_integration_tests()
