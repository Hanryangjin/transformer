import json
import os
from tqdm import tqdm

def combine_datasets():
    # ValidationData 디렉토리 경로
    val_data_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 결과를 저장할 리스트
    combined_data = []
    
    # 디렉토리 내의 모든 JSON 파일 처리
    json_files = [f for f in os.listdir(val_data_dir) if f.endswith('.json')]
    
    print("검증 데이터셋 통합 시작...")
    for json_file in tqdm(json_files, desc="파일 처리 중"):
        file_path = os.path.join(val_data_dir, json_file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 데이터가 'data' 키를 가지고 있는지 확인
            if 'data' not in data:
                print(f"경고: {json_file}에 'data' 키가 없습니다.")
                continue
            
            # 각 항목에서 err_sentence와 cor_sentence 추출
            for item in data['data']:
                if 'annotation' not in item:
                    continue
                
                annotation = item['annotation']
                if 'err_sentence' not in annotation or 'cor_sentence' not in annotation:
                    continue
                
                combined_data.append({
                    'err_sentence': annotation['err_sentence'],
                    'cor_sentence': annotation['cor_sentence']
                })
                
        except Exception as e:
            print(f"오류: {json_file} 처리 중 문제 발생: {str(e)}")
            continue
    
    # 결과를 새로운 JSON 파일로 저장
    output_data = {'data': combined_data}
    output_path = os.path.join(val_data_dir, 'combined_validation_dataset.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n검증 데이터셋 통합 완료!")
    print(f"총 {len(combined_data)}개의 문장 쌍이 추출되었습니다.")
    print(f"결과가 {output_path}에 저장되었습니다.")

if __name__ == '__main__':
    combine_datasets() 