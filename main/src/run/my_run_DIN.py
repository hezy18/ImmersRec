import subprocess

commend_list=['python','main.py', '--model_name', 'DIN', '--dataset', 'KuaiRand_history_test'
              ,'--include_item_features','1'
              ,'--include_context_features','1'
              ,'--include_immersion','1'
              ,'--include_source_domain','1'
              ,'--pretrained','1'
              ,'--fixed_no','1'
              ,'--DANN','1'
              ,'--gpu','1'] 
# subprocess.run(commend_list)

def run_DIN_ImmersRec(script_content, dataset):
    # embedding_list=['64','32']
    # lr_list = ['0.01','0.001','0.0001']
    # att_list=['\'[32]\'','\'[64]\'','\'[64,64]\'']
    wDA_list = ['0.1','0.01','0.001']
    wClf_list = ['0.1','0.01','0.001']
    att_list=['\'[64]\'','\'[64,64]\'','\'[64,64,64]\'']
    dnn_list=['\'[64]\'','\'[64,64]\'','\'[64,64,64]\'']
    for att in att_list:
        for dnn in dnn_list:
            for wDA in wDA_list:
                for wClf in wClf_list:
                    commend_list=['python','main.py', '--model_name', 'DIN', '--dataset', dataset
                                ,'--att_layers',att, '--dnn_layers', dnn, '--history_max', '20' 
                                ,'--tradeoff_DA', wDA,'--tradeoff_Clf', wClf
                                ,'--include_item_features','1','--include_context_features','1','--include_immersion','1'
                                ,'--include_source_domain','1','--pretrained','1','--fixed_no','1','--DANN','1'
                                ,'--eval_batch_size','8'
                                # ,'--gpu', '1'
                                ]
                    # subprocess.run(commend_list)
                    content = ' '.join(commend_list)  
                    script_content += content 
                    script_content += '\n'
    return script_content

script_content=''
script_content = run_DIN_ImmersRec(script_content,'KuaiRand_history')
    
with open('run_Kuai_DIN_ImmersRec.sh','w') as f:
    f.write(script_content)
