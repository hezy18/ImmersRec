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
    
def run_DIN_baseline(script_content):
    att_list=['\'[64]\'','\'[64,64]\'','\'[64,64,64]\'']
    dnn_list=['\'[64]\'','\'[64,64]\'','\'[64,64,64]\'']
    for att in att_list:
        for dnn in dnn_list:
            commend_list=['python','main.py', '--model_name', 'SDIM', '--dataset', 'KuaiRand_history'
            ,'--att_layers',att, '--dnn_layers', dnn, '--history_max', '20' 
            ,'--include_item_features','1','--include_context_features','1','--include_immersion','0'
            ,'--eval_batch_size','8'
            # ,'--gpu','2'
            ]
            # subprocess.run(commend_list)
            content = ' '.join(commend_list)  
            script_content += content 
            script_content += '\n'
    return script_content

script_content = ''
script_content = run_DIN_baseline(script_content)


with open('run_Kuai_DIN_baseline.sh','w') as f:
    f.write(script_content)