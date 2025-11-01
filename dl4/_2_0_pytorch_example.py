# pytorch_example.py
import torch

print("--- PyTorch 예제 ---")
print(f"PyTorch 버전: {torch.__version__}")

# 1. 텐서(Tensor) 생성
# torch.tensor는 텐서를 생성합니다.
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
print(f"PyTorch 텐서 a:\n{a}")

# 2. 텐서 연산
c = torch.matmul(a, b)
print(f"행렬 곱셈 결과 c:\n{c}")

# 3. 자동 미분 (Autograd)
# requires_grad=True로 설정하여 미분 연산을 추적하도록 명시합니다.
x = torch.tensor(3.0, requires_grad=True)
y = x * x
y.backward()  # 역전파(Backpropagation) 실행

print(f"dy/dx (x=3) = {x.grad}")  # 2 * 3 = 6.0

# 4. 간단한 모델 구축 및 학습
# torch.nn.Sequential을 사용해 모델을 순차적으로 구성합니다.
model = torch.nn.Sequential(
    torch.nn.Linear(in_features=2, out_features=4),  # 입력 층
    torch.nn.Linear(in_features=4, out_features=2),  # 출력 층
)
print("\nPyTorch 모델 구조:")
print(model)
