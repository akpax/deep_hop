# adjust path so it can find app.py
import sys
sys.path.insert(0, '/home/ec2-user/deep_hop')



import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_route(client):
    """Test the home route."""
    response = client.get('/')
    assert response.status_code == 200
    
def test_load_model_route(client):
    """Test load model route succesully loads GPTJ model """
    response = client.get("/display")
    assert response.status_code == 200
