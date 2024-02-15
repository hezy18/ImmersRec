import subprocess

def run_DCN_baseline(script_content):
    cross_num_list = ['1','3','6']
    layer_list= ['\'[64]\'','\'[128]\'','\'[64,64]\'']
    for cross_num in cross_num_list:
        for layer in layer_list:
            commend_list=['python','main.py', '--model_name', 'DCN', '--dataset', 'KuaiRand_history'
            ,'----attention_size',att_size, '--num_heads',num_heads
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

script_content = ''
script_content = run_DCN_baseline(script_content)


with open('run_Kuai_DCN_baseline.sh','w') as f:
    f.write(script_content)