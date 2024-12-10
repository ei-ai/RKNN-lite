# data_preprocessing.py

import os

def load_dataset(file_path):
    """
    토큰화된 데이터셋을 로드합니다.
    :param file_path: 데이터셋 파일 경로
    :return: 2차원 리스트 형태로 반환된 데이터셋
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 각 줄을 토큰 리스트로 변환
    data = [list(map(int, line.strip().split())) for line in lines]
    return data


def adjust_input_size(data, required_size, padding_value=0):
    """
    입력 데이터를 모델의 요구사항에 맞게 크기를 조정합니다.
    :param data: 2차원 리스트 형태의 입력 데이터
    :param required_size: 요구되는 입력 크기
    :param padding_value: 패딩에 사용할 값 (기본값: 0)
    :return: 크기가 조정된 데이터셋
    """
    adjusted_data = []
    for sample in data:
        if len(sample) < required_size:
            # 패딩 추가
            sample = sample + [padding_value] * (required_size - len(sample))
        elif len(sample) > required_size:
            # 샘플 자르기
            sample = sample[:required_size]
        adjusted_data.append(sample)
    return adjusted_data


if __name__ == "__main__":
    # 테스트용 코드
    SRC_DATASET = './srcdataset.txt'
    REQUIRED_SIZE = 5120

    print("데이터 로드 중...")
    src_data = load_dataset(SRC_DATASET)

    print("입력 크기 조정 중...")
    src_data = adjust_input_size(src_data, REQUIRED_SIZE)

    print(f"샘플 수: {len(src_data)}")
    print(f"첫 번째 샘플: {src_data[0]}")


