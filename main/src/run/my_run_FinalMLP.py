import subprocess

commend_list=['python','main.py', '--model_name', 'FinalMLP', '--dataset', 'KuaiRand_history_test'
              ,'--num_heads','1', '--fs1_context','["c_immersion","c_session_order"]', '--fs2_context','["item_id"]'
              ,'--include_item_features','1','--include_context_features','1' ,'--include_immersion','1'
              ,'--include_source_domain','1', '--pretrained', '1' ,'--fixed_no', '1', '--DANN','1'
            #   ,'--gpu','2'
              ] 
# subprocess.run(commend_list)

def run_FinalMLP_ImmersRec(script_content):
    wDA_list = ['0','0.1','0.01','0.001']
    wClf_list = ['0','0.1','0.01','0.001']
    num_heads_list = ['1','2','4'] 
    for fs1 in ['\'["c_immersion","c_session_order"]\'','\'["c_immersion"]\'','\'[]\'']:
        for fs2 in ['\'["user_id","item_id"]\'','\'["item_id"]\'','\'["user_id"]\'', '\'[]\'']: 
            for num_heads in num_heads_list:
                for wDA in wDA_list:
                    for wClf in wClf_list:
                        commend_list=['python','main.py', '--model_name', 'FinalMLP', '--dataset', 'KuaiRand_history'
                        ,'--tradeoff_DA', wDA,'--tradeoff_Clf', wClf
                        ,'--num_heads',num_heads, '--fs1_context',fs1, '--fs2_context',fs2
                        ,'--include_item_features','1','--include_context_features','1','--include_immersion','1'
                        ,'--include_source_domain','1','--pretrained','1','--fixed_no','1','--DANN','1'
                        ,'--eval_batch_size','8'
                        # ,'--gpu','2'
                        ] 
                        subprocess.run(commend_list)
                        content = ' '.join(commend_list)  
                        script_content += content 
                        script_content += '\n'
    return script_content

script_content = ''
script_content = run_FinalMLP_ImmersRec(script_content)


with open('run_FinalMLP_ImmersRec.sh','w') as f:
    f.write(script_content)
