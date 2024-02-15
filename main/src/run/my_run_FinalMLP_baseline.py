import subprocess
    
def run_FinalMLP_baseline(script_content):
    num_heads_list = ['1','2','4'] 
    for fs1 in ['\'["c_session_order"]\'','\'[]\'']:
        for fs2 in ['\'["user_id","item_id"]\'','\'["item_id"]\'','\'["user_id"]\'', '\'[]\'']: 
            commend_list=['python','main.py', '--model_name', 'FinalMLP', '--dataset', 'KuaiRand_history'
                        ,'--num_heads',num_heads, '--fs1_context',fs1, '--fs2_context',fs2
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


script_content = run_FinalMLP_baseline(script_content)

with open('run_FinalMLP_baseline.sh','w') as f:
    f.write(script_content)