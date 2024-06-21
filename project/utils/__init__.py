import os
import re
import torch
from torch import nn
from PIL import Image
from torchvision import models, transforms
from transformers import PreTrainedModel, PretrainedConfig, AutoModelForSequenceClassification, AutoTokenizer, ViTForImageClassification, ViTImageProcessor

class MultiModalConfig(PretrainedConfig):
    model_type = "multimodal"

    def __init__(self, resnet_model_paths=None, vit_model_paths=None, nlp_transformers_model_paths=None, **kwargs):
        super().__init__(**kwargs)
        self.resnet_model_paths = resnet_model_paths
        self.vit_model_paths = vit_model_paths
        self.nlp_transformers_model_paths = nlp_transformers_model_paths
        self.class_names = ["PG", "PG-13", "R"]
        self.label2id = {label: i for i, label in enumerate(self.class_names)}
        self.id2label = {i: label for label, i in self.label2id.items()}

class MultiModalModel(PreTrainedModel):
    config_class = MultiModalConfig

    def __init__(self, config: MultiModalConfig):
        super().__init__(config)
        self.num_classes = 3
        
        # Initialize the models and set requires_grad = False to freeze them
        self.resnet_models = nn.ModuleDict()
        if config.resnet_model_paths:
            for model_id, model_path in config.resnet_model_paths.items():
                model = models.resnet50() if "resnet50" in model_id else models.resnet18()
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, self.num_classes)
                model.load_state_dict(torch.load(f"{model_path}/best_model_params.pt", map_location='cpu'))
                model.fc = nn.Linear(num_ftrs, num_ftrs//4)
                for param in model.parameters():
                    param.requires_grad = False
                self.resnet_models[model_id] = model

        self.vit_models = nn.ModuleDict()
        if config.vit_model_paths:
            for model_id, model_path in config.vit_model_paths.items():
                model = ViTForImageClassification.from_pretrained(model_path)
                for param in model.parameters():
                    param.requires_grad = False
                self.vit_models[model_id] = model

        self.nlp_transformers_models = nn.ModuleDict()
        if config.nlp_transformers_model_paths:
            for model_id, model_path in config.nlp_transformers_model_paths.items():
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                for param in model.parameters():
                    param.requires_grad = False
                self.nlp_transformers_models[model_id] = model

        self.fc = nn.Linear(self.get_feature_size(), self.num_classes)

    @staticmethod
    def clean_text(text: str) -> str:
        """
        This method cleans prompt data, removing extraneous punctuation meant to denote blending, loras, or models without removing names or tags.
        We also get rid of extraneous spaces or line breaks to reduce tokens and maintain as much semantic logic as possible
        """
        text = str(text)
        # Remove additional characters: ( ) : < > [ ]
        cleaned_text = re.sub(r"[():<>[\]]", " ", text)
        cleaned_text = cleaned_text.replace("\n", " ")
        # Replace multiple spaces with a single space
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)
        cleaned_text = re.sub(r"\s*,\s*", ", ", cleaned_text)

        return cleaned_text.strip()

    def get_classifier_in_features(self, model):
        """
        Function to get the number of input features to the classifier layer.
        Handles different model structures such as BERT and RoBERTa.
        """
        try:
            # RoBERTa's classifier
            return model.classifier.out_proj.in_features
        except AttributeError:
            pass

        try:
            # BERT's classifier
            return model.classifier.linear.in_features
        except AttributeError:
            pass

        try:
            # DistilBERT's classifier
            return model.classifier.in_features
        except AttributeError:
            pass

        try: 
            #ViT's classifier
            return model.classifier.in_features
        except AttributeError:
            pass

        try: 
            return model.fc.in_features
        except AttributeError:
            pass

        # Add other model types as needed
        raise ValueError("Unknown model structure or unsupported model type.")

    def get_feature_size(self):
        feature_size = 0
        for model in self.resnet_models.values():
            feature_size += model.fc.out_features

        for model in self.vit_models.values():
            feature_size += model.config.hidden_size

        for model in self.nlp_transformers_models.values():
            feature_size += model.config.hidden_size

        return feature_size

    def forward(self, image: Image, text: str):
        features = []

        # Extract features from ResNet models
        for model in self.resnet_models.values():
            image_tensor = self.preprocess_image(image)
            with torch.no_grad():
                features.append(model(image_tensor))

        # Extract features from ViT models
        for model in self.vit_models.values():
            processor = ViTImageProcessor.from_pretrained(model.config._name_or_path)
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states = True)
            hidden_states = outputs.hidden_states[-1]
            features.append(hidden_states[:,0, :])

        # Extract features from nlp_transformers models
        for model in self.nlp_transformers_models.values():
            tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
            clean_text = self.clean_text(text)
            inputs = tokenizer(clean_text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states = True)
            hidden_states = outputs.hidden_states[-1]
            features.append(hidden_states[:,0,:])

        features = torch.cat(features, dim=1)
        print(features.shape)
        output = self.fc(features)
        return output

    def preprocess_image(self, image):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)

    @classmethod
    def from_pretrained(cls, model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config_path = model_name_or_path if os.path.isdir(model_name_or_path) else None
            config = cls.config_class.from_pretrained(model_name_or_path, cache_dir=kwargs.get("cache_dir"), config_path=config_path)
        return cls(config)

    def save_pretrained(self, save_directory):
        super().save_pretrained(save_directory)
        self.config.save_pretrained(save_directory)


