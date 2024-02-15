import subprocess
    
def run_DeepFM_baseline(script_content):
    layer_list= ['\'[64]\'','\'[128]\'','\'[64,64]\'']
    for layer in layer_list:
        commend_list=['python','main.py', '--model_name', 'DeepFM', '--dataset', 'KuaiRand_history'
                    ,'--layers',layer
                    ,'--include_item_features','1','--include_context_features','1','--include_immersion','0'
                    ,'--include_source_domain','0','--pretrained','0','--fixed_no','0','--DANN','0'
                    ,'--eval_batch_size','8'
                    # ,'--gpu','2'
                    ]
        # subprocess.run(commend_list)
        content = ' '.join(commend_list)  
        script_content += content 
        script_content += '\n'
    return script_content


script_content = run_DeepFM_baseline(script_content)

with open('run_Kuai_DeepFM_baseline.sh','w') as f:
    f.write(script_content)
    
