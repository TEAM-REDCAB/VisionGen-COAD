import MSIpred as mp

# maf 파일 읽어오기
test_maf = mp.Raw_Maf(maf_path = "~/data/TCGA_COAD_multimodal_data/TCGA_COAD.wxs.common_parsed.maf")

# http://hgdownload.cse.ucsc.edu/goldenPath/hg38/database/simpleRepeat.txt.gz 로부터 다운받은 simpleRepeat.txt 파일을 이용하여 tagged maf 파일 생성
test_maf.create_tagged_maf(ref_repeats_file='/home/team1/lhj/VisionGen-COAD/lhj/raw/simpleRepeat.txt',tagged_maf_file = 'tagged_maf')

# tagged maf 파일로부터 feature table 생성 및 MSI 예측
tagged_maf = mp.Tagged_Maf(tagged_maf_path='tagged.maf')
tagged_features = tagged_maf.make_feature_table(exome_size=44)
predicted_MSI = mp.msi_prediction(feature_table=tagged_features,svm_model=None)
predicted_MSI["patient_id"]=predicted_MSI["Tumor"].str.split("-").str[0:3].str.join("-")

MSIpred_result=predicted_MSI[["patient_id","Predicted_MSI_Status"]].drop_duplicates()

MSIpred_result.replace("MSI-H", "MSIMUT", inplace=True)
MSIpred_result.columns=["patient","type"]
MSIpred_result.to_csv("MSIpred_result.tsv",index=False,sep="\t")