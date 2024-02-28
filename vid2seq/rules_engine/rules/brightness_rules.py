import numpy as np
import cv2

from .abstract_rules import AbstractRules 

class BrightnessRules(AbstractRules):
    def __init__(self, kwargs):
        super().__init__(kwargs)
        self.video_images_folder = kwargs["video_images_folder"]
        self.color_intensity_thres = kwargs["color_intensity_threshold"]
    
    def brightness_rule(self, image_path, sentence):
        # "bright day" -> "dark day", "bright environment -> dark environment", "bright brightness -> dark brightness",  
        # "bright surroundings" -> "dark surroundings" , "bright lighting conditions" -> "dark lighting conditions" 
        # "bright outside" -> "dark outside", "bright conditions -> dark conditions" -> "bright visibility" -> "dark visibility"

        # "clear and bright" -> "clear and dark"
        # "Despite the brightness" -> "Despite the darkness"
        # "Despite the surroundings being bright" -> "Despite the surroundings being dark"

        # "The environment is/was bright"
        # "The brightness is/was bright" -> "The brightness is/was dark"
        # "The surrounding environment is/was bright" -> "The surrounding environment is/was dark"
        # "the brightness of the surroundings is/was bright" -> "the brightness of the surroundings is/was dark"
        # "The lightning conditions are/were bright" -> "The lightning conditions are/were dark"

        src_img = cv2.imread(image_path)
        average_color_row = np.average(src_img, axis=0)
        average_color = np.average(average_color_row, axis=0)

        set_of_rules = [
            ("bright day","dark day"), ("bright environment, dark environment"), ("bright brightness,dark brightness"),  
            ("bright surroundings", "dark surroundings") , ("bright lighting conditions", "dark lighting conditions"), 
            ("bright outside", "dark outside") , ("bright conditions, dark conditions"), ("bright visibility", "dark visibility"),

            ("clear and bright", "clear and dark"),
            ("Despite the brightness", "Despite the darkness"),
            ("Despite the surroundings being bright", "Despite the surroundings being dark"),
            ("In the bright ", "In the dark "), ("in the bright ", "in the dark ")

            ("The environment is bright", "The environment is dark"), ("The environment is bright", "The environment is dark") ,
            ("The brightness is bright", "The brightness is dark"), ("The brightness was bright", "The brightness was dark"),
            ("The surrounding environment is bright", "The surrounding environment is dark"), ("The surrounding environment was bright", "The surrounding environment was dark"),
            ("The brightness of the surroundings is bright", "the brightness of the surroundings is dark"), ("The brightness of the surroundings was bright","the brightness of the surroundings was dark"),
            ("The lightning conditions are bright", "The lightning conditions are dark"), ("The lightning conditions were bright", "The lightning conditions were dark")
        ]
        
        if np.average(average_color, axis=0) <= self.color_intensity_thres:
            for old, new in set_of_rules:
                sentence = sentence.replace(old, new)
                
        return sentence


    def execute(self, preds: str):
        new_preds = {}
        for key in preds:
            video_name , _ = key.split("#")
            image_path = f"{self.video_images_folder}/{video_name}.png"
            sentence = preds[key]
            
            new_sentence = self.brightness_rule(image_path=image_path, sentence=sentence)

            if key not in new_preds:
                new_preds[key] = new_sentence
            
        return new_preds