# adjust path so it can find model_manager.py
import sys
sys.path.insert(0, '/home/ec2-user/deep_hop')
from model_manager import GPTJ

import pytest


# setting scope to class tears down fixture at the end of last test in class, so we dont need to reload for each test
@pytest.fixture(scope="class") 
def model():
    model = GPTJ("model/gpt-j-6b.bin")
    return model

class TestModelManager():
    def test_model_laoding(self,model):
        assert model is not None
        
    def test_generate_verse(self,model):
        """Tests that model retruns list"""
        verses = model.generate_verse("this is a test verse, yee")
        assert type(verses) == list
        
    def test_has_characters_none(self,model):
        """Tests model filtering function for removing blacnk space"""
        test_string_empty = "          "
        assert not model.has_characters(test_string_empty)
        
    def test_has_characters_exists(self,model):
        """Tests model filtering function for removing blacnk space"""
        test_string_empty = "thisa is not empty"
        assert model.has_characters(test_string_empty)
        
    
    
    
    
    