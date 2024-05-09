

# #====API NAME ONLT===
#
#
#
#
#==
#
# from flask import Flask, request, jsonify
# from flask_restful import Api, Resource
# from torchvision.models.resnet import resnet50
# import torch.nn as nn
# import torch.nn.functional as F
# from PIL import Image
# import torch
# import numpy as np
# import os
# import mysql.connector
# from werkzeug.utils import secure_filename
# import mysql.connector


# app = Flask(__name__)
# api = Api(app)

# class Predict(Resource):
#     def __init__(self):
#         super(Predict, self).__init__()

#         # Load the model checkpoint and create the model
#         self.model_path = './models/40model.pth'
#         self.model = resnet50(pretrained=False)
#         self.model.fc = torch.nn.Sequential(
#             torch.nn.Dropout(0.5),
#             torch.nn.Linear(2048, 40)
#         )
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
#         self.model.eval()

#         # Classes for predictions
#         # self.classes = ['French Fries', 'Fried Calamari', 'Fried Rice', 'Greek Salad', 'Hot Dog', 'Ice Cream', 'Mussels',
#         #                 'Onion Rings', 'Oysters', 'Panna Cotta', 'Peking Duck', 'Pho', 'Prime Rib', 'Red Velvet Cake',
#         #                 'Spaghetti Carbonara', 'Cup Cakes', 'Dumplings', 'Seaweed Salad', 'Waffles', 'Deviled Eggs',
#         #                 'Escargots', 'Macaroni and Cheese', 'Samosa', 'Croque Madame', 'Pizza', 'Guacamole',
#         #                 'Chicken Quesadilla', 'Baked Potato', 'Taquito', 'Hamburger', 'Macarons', 'Donuts', 'Takoyaki',
#         #                 'Dhokla', 'Kulfi', 'Idli', 'Pav Bhaji', 'Chai', 'Chole Bhature', 'Biryani']
#         self.classes =  ['french_fries', 'fried_calamari', 'fried_rice', 'greek_salad', 'hot_dog', 'ice_cream', 'mussels', 'onion_rings', 'oysters', 'panna_cotta', 'peking_duck', 'pho', 'prime_rib', 'red_velvet_cake', 'spaghetti_carbonara', 'cup_cakes', 'dumplings', 'seaweed_salad', 'waffles', 'deviled_eggs', 'escargots', 'macaroni_and_cheese', 'samosa', 'croque_madame', 'pizza', 'guacamole', 'chicken_quesadilla', 'baked_potato', 'Taquito', 'hamburger', 'macarons', 'donuts', 'takoyaki', 'dhokla', 'kulfi', 'idli', 'pav_bhaji', 'chai', 'chole_bhature', 'Biryani']

#         self.classes.sort()

#         # Connect to the MySQL database
#         self.db_connection = mysql.connector.connect(
#             host="localhost",
#         user = "root",
#         password = "anas",
#         database = "food_recipe"
#         )
#         self.db_cursor = self.db_connection.cursor()

#     def __del__(self):
#         # Close the database connection when the Predict object is deleted
#         if self.db_connection.is_connected():
#             self.db_cursor.close()
#             self.db_connection.close()

#     def post(self):
#         try:
#             # Get the image file from the request
#             img_file = request.files['image']

#             filename = secure_filename(img_file.filename)
#             img_file.save(os.path.join("uploads", filename))

#             # Process the image
#             img_path = './uploads/' + filename
#             img = self.process_image(img_path)

#             # Make predictions
#             top_prob, top_class = self.predict(img, self.model)
#             if top_prob >= 90:
#             # Get the predicted class
#               if top_class in range(0, 40):
#                 predicted_title = self.classes[top_class]
#                 recipe_details = self.get_recipe_details(predicted_title)
#               else:
#                 predicted_title = "non_food"
#                 recipe_details = None

#             # Return the prediction and recipe details
#               return jsonify({'recipe_details': recipe_details, 'predicted_title': predicted_title})
#             else:
#              return jsonify({'error': 'Picture not detected'})

#         except Exception as e:
#               return jsonify({'error': str(e)})

#         #
#         #     # Get the predicted class
#         #     if top_class in range(0, 40):
#         #         predicted_title = self.classes[top_class]
#         #         recipe_details = self.get_recipe_details(predicted_title)
#         #     else:
#         #         predicted_title = "non_food"
#         #         recipe_details = None
#         #
#         #     # Return the prediction and recipe details
#         #     return jsonify({'recipe_details': recipe_details, 'predicted_title': predicted_title})
#         #
#         #
#         # except Exception as e:
#         #     return jsonify({'error': str(e)})

#     def get_recipe_details(self, predicted_title):
#         try:
#             # Query the database for all recipe details based on the predicted title
#             query = f"SELECT * FROM recipes WHERE title = '{predicted_title}'"
#             self.db_cursor.execute(query)
#             recipes = self.db_cursor.fetchall()

#             if recipes:
#                 recipe_details_list = []
#                 for recipe in recipes:
#                     recipe_details = {
#                         'title': recipe[1],
#                         'ingredients': recipe[2],
#                         'instructions': recipe[3]
#                     }
#                     recipe_details_list.append(recipe_details)
#                 return recipe_details_list
#             else:
#                 return None
#         except Exception as e:
#             return None

#     def process_image(self, image_path):
#         img = Image.open(image_path)
#         width, height = img.size
#         img = img.resize((255, int(255 * (height / width))) if width < height else (int(255 * (width / height)), 255))
#         width, height = img.size
#         left = (width - 224) / 2
#         top = (height - 224) / 2
#         right = (width + 224) / 2
#         bottom = (height + 224) / 2
#         img = img.crop((left, top, right, bottom))
#         img = np.array(img)
#         img = img.transpose((2, 0, 1))
#         img = img / 255
#         img[0] = (img[0] - 0.485) / 0.229
#         img[1] = (img[1] - 0.456) / 0.224
#         img[2] = (img[2] - 0.406) / 0.225
#         img = img[np.newaxis, :]
#         img = torch.from_numpy(img)
#         img = img.float()
#         return img

#     def predict(self, image, model):
#         image = image.to(self.device)
#         output = model(image)
#         output = torch.exp(output)
#         probs, classes = output.topk(1, dim=1)
#         return 100 if probs.item() > 100 else probs.item(), classes.item()

# api.add_resource(Predict, '/predict')

# if __name__ == '__main__':
#      app.run(debug=True)


from flask import Flask, request, jsonify
import torch
from flask_restful import Api, Resource
from torchvision.models.resnet import resnet50
from PIL import Image
import os
import numpy as np
import mysql.connector
from werkzeug.utils import secure_filename

app = Flask(__name__)
api = Api(app)

class Predict(Resource):
    def __init__(self):
        super(Predict, self).__init__()

        # Load the model checkpoint and create the model
        self.model_path = './models/40model.pth'
        self.model = resnet50(pretrained=False)
        self.model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(2048, 40)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

        # Classes for predictions
        self.classes = ['french_fries', 'fried_calamari', 'fried_rice', 'greek_salad', 'hot_dog', 'ice_cream', 'mussels', 'onion_rings', 'oysters', 'panna_cotta', 'peking_duck', 'pho', 'prime_rib', 'red_velvet_cake', 'spaghetti_carbonara', 'cup_cakes', 'dumplings', 'seaweed_salad', 'waffles', 'deviled_eggs', 'escargots', 'macaroni_and_cheese', 'samosa', 'croque_madame', 'pizza', 'guacamole', 'chicken_quesadilla', 'baked_potato', 'Taquito', 'hamburger', 'macarons', 'donuts', 'takoyaki', 'dhokla', 'kulfi', 'idli', 'pav_bhaji', 'chai', 'chole_bhature', 'Biryani']
        self.classes.sort()

        # Connect to the MySQL database
        self.db_connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="anas",
            database="food_recipe"
        )
        self.db_cursor = self.db_connection.cursor()

    def __del__(self):
        # Close the database connection when the Predict object is deleted
        if self.db_connection.is_connected():
            self.db_cursor.close()
            self.db_connection.close()

    def post(self):
        try:
            # Get the image file from the request
            img_file = request.files['image']

            filename = secure_filename(img_file.filename)
            img_file.save(os.path.join("uploads", filename))

            # Process the image
            img_path = './uploads/' + filename
            img = self.process_image(img_path)

            # Make predictions
            top_prob, top_class = self.predict(img, self.model)

            # Get the predicted class
            if top_class in range(0, 40):
                predicted_title = self.classes[top_class]
                recipe_details = self.get_recipe_details(predicted_title)
            else:
                predicted_title = "non_food"
                recipe_details = None

            # Return the prediction and recipe details
            if recipe_details:
                return jsonify({'recipe_details': recipe_details})
            else:
                return jsonify({'predicted_title': predicted_title })

        except Exception as e:
            return jsonify({'error': str(e)})

    def get_recipe_details(self, predicted_title):
        try:
            # Query the database for all recipe details based on the predicted title
            query = f"SELECT * FROM recipes WHERE title = '{predicted_title}'"
            self.db_cursor.execute(query)
            recipes = self.db_cursor.fetchall()

            if recipes:
                recipe_details_list = []
                for recipe in recipes:
                    recipe_details = {
                        'title': recipe[1],
                        'ingredients': recipe[2],
                        'instructions': recipe[3]
                    }
                    recipe_details_list.append(recipe_details)
                return recipe_details_list
            else:
                return None
        except Exception as e:
            return None

    def process_image(self, image_path):
        img = Image.open(image_path)
        width, height = img.size
        img = img.resize((255, int(255 * (height / width))) if width < height else (int(255 * (width / height)), 255))
        width, height = img.size
        left = (width - 224) / 2
        top = (height - 224) / 2
        right = (width + 224) / 2
        bottom = (height + 224) / 2
        img = img.crop((left, top, right, bottom))
        img = np.array(img)
        img = img.transpose((2, 0, 1))
        img = img / 255
        img[0] = (img[0] - 0.485) / 0.229
        img[1] = (img[1] - 0.456) / 0.224
        img[2] = (img[2] - 0.406) / 0.225
        img = img[np.newaxis, :]
        img = torch.from_numpy(img)
        img = img.float()
        return img

    def predict(self, image, model):
        image = image.to(self.device)
        output = model(image)
        output = torch.exp(output)
        probs, classes = output.topk(1, dim=1)
        return 100 if probs.item() > 100 else probs.item(), classes.item()


api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run(debug=True)
