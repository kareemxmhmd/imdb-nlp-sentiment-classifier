import pytest
from src.data import clean_text
import pandas as pd

def test_clean_text():
    html_text = "This is a <br> test with <b>HTML</b>"
    assert clean_text(html_text) == "this is a   test with  html "
    
    punct_text = "Hello, world! How's it going?"
    assert clean_text(punct_text) == "hello world hows it going"
    
    mixed_text = "<h1>Great movie!</h1> 10/10."
    assert clean_text(mixed_text) == " great movie   1010"
