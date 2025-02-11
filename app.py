import torch
import torch.nn as nn
import os
from flask import Flask, request, jsonify, render_template
from PIL import Image
import torchvision.transforms as transforms

app = Flask(__name__)

#Define PatchEmbed Class
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

#  Attention Block
class Attention(nn.Module):
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) 
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        n_samples, n_tokens, dim = x.shape
        if dim != self.head_dim * self.n_heads:
            raise ValueError("Dimension mismatch in Attention layer!")

        qkv = self.qkv(x).reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(n_samples, n_tokens, dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)  #  Matches DeiT
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features) 
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x) 
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x) 
        x = self.drop(x)
        return x


# Added Transformer Block✅
class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=hidden_features, out_features=dim, drop=drop)  #  updated MLP

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  
        x = x + self.mlp(self.norm2(x))  
        return x

#Costom Vision Transformer Model
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, n_classes=1000, embed_dim=768, depth=12, n_heads=12, mlp_ratio=4.):  
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.pos_drop = nn.Dropout(p=0.1)
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(n_samples, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        return x

# Loading Pretrained Weights into Custom Vision Transformer Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformer(img_size=224, patch_size=16, in_c=3, n_classes=1000, embed_dim=768, depth=12, n_heads=12).to(device)

pretrained_model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
pretrained_weights = pretrained_model.state_dict()


missing_keys, unexpected_keys = model.load_state_dict(pretrained_weights, strict=False)
print(f"✅ Weights Loaded Successfully! ✅\nMissing keys: {missing_keys}\nUnexpected keys: {unexpected_keys}")
model.eval()
#  Image Preprocessing (Same as DeiT Training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Match ImageNet stats
])
with open("imagenet_labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]  # Read all label




@app.route('/')
def home():
    return render_template('index.html')

#Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model is not loaded. Please train and save the model."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    try:
        # Process the image
        image = Image.open(file).convert('RGB')  # Convert to RGB
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
        
        # Run inference
        with torch.no_grad():
            output = model(image)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()  # Get class index

      
        predicted_label = class_names[predicted_class] if predicted_class < len(class_names) else "Unknown"

        return jsonify({"predicted_class": predicted_class, "label": predicted_label})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
