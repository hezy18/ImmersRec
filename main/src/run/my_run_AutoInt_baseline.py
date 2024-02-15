import subprocess

def run_AutoInt_baseline(script_content):
    att_size_list = ['32','64']
    num_heads_list = ['1','2','3','4','6']
    for att_size in att_size_list:
        for num_heads in num_heads_list:
            commend_list=['python','main.py', '--model_name', 'AutoInt', '--dataset', 'KuaiRand_history'
            ,'--attention_size',att_size, '--num_heads',num_heads
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
script_content = run_AutoInt_baseline(script_content)


with open('run_Kuai_AutoInt_baseline.sh','w') as f:
    f.write(script_content)