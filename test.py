import numpy as np
import platform
from transformers import BertTokenizer
from rknnlite.api import RKNNLite
from data_preprocessing import load_dataset, adjust_input_size


# RKNN 모델 파일 경로
RKNN_MODEL = 'transformer.rknn'

# 데이터셋 파일 경로
SRC_DATASET = './srcdataset.txt'
TGT_DATASET = './tgtdataset.txt'

# Device Tree Node
DEVICE_COMPATIBLE_NODE = '/proc/device-tree/compatible'

# Tokenizer 초기화
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

def get_host():
    """디바이스 정보를 가져옵니다."""
    system = platform.system()
    machine = platform.machine()
    os_machine = system + '-' + machine
    if os_machine == 'Linux-aarch64':
        try:
            with open(DEVICE_COMPATIBLE_NODE) as f:
                device_compatible_str = f.read()
                if 'rk3588' in device_compatible_str:
                    host = 'RK3588'
                elif 'rk3562' in device_compatible_str:
                    host = 'RK3562'
                else:
                    host = 'RK3566_RK3568'
        except IOError:
            print('디바이스 노드를 읽을 수 없습니다: {}'.format(DEVICE_COMPATIBLE_NODE))
            exit(-1)
    else:
        host = os_machine
    return host

def load_dataset(filepath):
    """텍스트 파일을 읽어 리스트로 반환합니다."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [list(map(int, line.strip().split())) for line in lines]

def show_top5(result):
    """
    Transformer 모델의 결과를 출력.
    result는 토큰 ID 리스트로 가정.
    """
    output_ids = result[0]  # 모델 결과는 리스트 형태로 반환됨
    decoded_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    print('\nTransformer 모델 결과:')
    print(f'출력된 문장: {decoded_text}')

if __name__ == '__main__':
    # RKNN 모델 로드
    print('--> Load RKNN model')
    rknn_lite = RKNNLite()
    ret = rknn_lite.load_rknn(RKNN_MODEL)
    if ret != 0:
        print('RKNN 모델 로드 실패')
        exit(ret)
    print('RKNN 모델 로드 완료.')

    # 데이터셋 로드
    print('--> 데이터셋 로드 중...')
    src_data = load_dataset(SRC_DATASET)
    src_data = adjust_input_size(src_data, REQUIRED_SIZE)
    tgt_data = load_dataset(TGT_DATASET)
    tgt_data = adjust_input_size(tgt_data, REQUIRED_SIZE)


    # 실행 환경 초기화
    print('--> Init runtime environment')
    ret = rknn_lite.init_runtime()
    if ret != 0:
        print('실행 환경 초기화 실패')
        exit(ret)
    print('실행 환경 초기화 완료.')

    # 추론 수행
    print('--> Running inference')
    for i, src_input in enumerate(src_data):
        # 입력 데이터 준비 (배치 크기 1)
        input_tensor = np.expand_dims(src_input, axis=0)

        # 추론 수행
        outputs = rknn_lite.inference(inputs=[input_tensor])

        # 결과 출력
        print(f'\n샘플 {i + 1}/{len(src_data)}')
        print('입력 텍스트:', tokenizer.decode(src_input, skip_special_tokens=True))
        show_top5(outputs)

    # RKNN 자원 해제
    rknn_lite.release()
    print('모든 작업 완료.')
