import requests
import pytest

# may need to modify clf reference and model path in apps.py as well
# may need to modify this path
from app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_client(client):
    r = client.get('/')
    print("\n====== Result TEST_CLIENT ======")
    print(r.data)
    print("TEST PASSED!")

def test_predict(client):
    r = client.get('/predict?age=5&absences=5&health=20')
    # loop through file with parameters and result and cross check
    print("\n====== Result TEST_PREDICT ======")
    print(int(r.data))
    assert(int(r.data) == 1)
    print("TEST PASSED!")