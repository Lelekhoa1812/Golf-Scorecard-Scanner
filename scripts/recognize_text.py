from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import cv2

def load_vietocr_model():
    """Load VietOCR model for text recognition."""
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = 'models/vietocr_weights.pth'
    config['device'] = 'cpu'
    config['predictor']['beamsearch'] = False
    return Predictor(config)

def recognize_text(image, model):
    """Recognize text from an image."""
    pil_image = Image.fromarray(image)
    return model.predict(pil_image)

if __name__ == "__main__":
    model = load_vietocr_model()
    image = cv2.imread("data/images/field.jpg")
    text = recognize_text(image, model)
    print(text)
