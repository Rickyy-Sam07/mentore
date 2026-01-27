"""
API Client for Remote DistilBERT Model
For Computer 2 (Streamlit App)
"""

import requests
from typing import Dict, List, Optional
import time


class RemoteModelClient:
    """Client to interact with remote DistilBERT API via ngrok"""
    
    def __init__(self, ngrok_url: str, timeout: int = 30):
        """
        Initialize the client
        
        Args:
            ngrok_url: The ngrok URL (e.g., https://abc123.ngrok.io)
            timeout: Request timeout in seconds
        """
        self.base_url = ngrok_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """Check if the API is healthy"""
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_info(self) -> Dict:
        """Get model information"""
        try:
            response = self.session.get(
                f"{self.base_url}/info",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predict(self, text: str, top_k: int = 5) -> Optional[Dict]:
        """
        Predict chapter from text
        
        Args:
            text: Input text
            top_k: Number of top predictions to return
            
        Returns:
            Prediction results or None if error
        """
        try:
            payload = {
                "text": text,
                "top_k": top_k
            }
            
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.Timeout:
            return {"error": "Request timeout - server not responding"}
        except requests.exceptions.ConnectionError:
            return {"error": "Connection error - check ngrok URL"}
        except requests.exceptions.HTTPError as e:
            return {"error": f"HTTP error: {e.response.status_code}"}
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}
    
    def test_connection(self) -> bool:
        """Test if connection to server is working"""
        try:
            health = self.health_check()
            return health.get("status") == "healthy"
        except:
            return False


# Example usage
if __name__ == "__main__":
    # Replace with your actual ngrok URL
    NGROK_URL = "https://YOUR-NGROK-URL.ngrok.io"
    
    print("Testing Remote Model Client...")
    print(f"Connecting to: {NGROK_URL}")
    
    client = RemoteModelClient(NGROK_URL)
    
    # Test connection
    print("\n1. Testing connection...")
    if client.test_connection():
        print("   Connection successful!")
    else:
        print("   Connection failed!")
        exit(1)
    
    # Get model info
    print("\n2. Getting model info...")
    info = client.get_info()
    print(f"   Model: {info.get('model_name')}")
    print(f"   Classes: {info.get('num_classes')}")
    
    # Make prediction
    print("\n3. Testing prediction...")
    text = "O Sage, I would hear of such a man from you, who art able to describe him to me."
    result = client.predict(text)
    
    if result and 'error' not in result:
        print(f"   Predicted: {result['predicted_chapter']}")
        print(f"   Confidence: {result['confidence']:.2%}")
    else:
        print(f"   Error: {result.get('error')}")
