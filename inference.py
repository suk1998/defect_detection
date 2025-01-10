import torch
from timm import create_model
from torch.utils.data import DataLoader
from defect_detection import inference, CustomDataset , getCAM
from torchvision import transforms as T
 # 인퍼런스 모듈 임포트

# 디바이스 설정 (GPU 사용 가능 여부 확인)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# 데이터셋 경로와 변환 정의
root = "C:/CODE/defect_detection/test_data"  # 데이터 경로 수정 필요
im_size = 224
transform = T.Compose([
    T.Resize((im_size, im_size)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 테스트 데이터셋 생성 및 로드
test_dataset = CustomDataset(root=root, transformations=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 클래스 이름 로드 (예시: 클래스 이름을 데이터셋에서 추출)
classes = test_dataset.cls_names  # {'defect1': 0, 'defect2': 1, ...}
print(f"Classes: {classes}")

# RexNet 모델 로드 및 설정
model = create_model("rexnet_150", pretrained=True, num_classes=len(classes))
model.load_state_dict(torch.load("saved_models/model_epoch100.pth"))
model.eval()

# 마지막 Conv 레이어와 FC 레이어 가중치 추출
final_conv = model.features[-1]
fc_params = list(model.head.fc.parameters())

# 인퍼런스 실행
inference(
    model=model.to(device), 
    device=device, 
    test_dl=test_loader, 
    num_ims=20, 
    row=4, 
    final_conv=final_conv, 
    fc_params=fc_params, 
    cls_names=list(classes.keys())
)

