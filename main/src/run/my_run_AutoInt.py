import subprocess

commend_list=['python','main.py', '--model_name', 'AutoInt', '--dataset', 'KuaiRand_history_test'
              ,'--include_item_features','1'
              ,'--include_context_features','1'
              ,'--include_immersion','1'
              ,'--include_source_domain','1'
              ,'--pretrained','1'
              ,'--fixed_no','1'
              ,'--DANN','1'
              ,'--gpu','2'] 
# subprocess.run(commend_list)

def run_AutoInt_ImmersRec(script_content):
    wDA_list = ['0','0.1','0.01','0.001']
    wClf_list = ['0','0.1','0.01','0.001']
    att_size_list = ['32','64']
    num_heads_list = ['1','2','3','4','6']
    for att_size in att_size_list:
        for num_heads in num_heads_list:
            for wDA in wDA_list:
                for wClf in wClf_list:
                    commend_list=['python','main.py', '--model_name', 'AutoInt', '--dataset', 'KuaiRand_history'
                    ,'--tradeoff_DA', wDA,'--tradeoff_Clf', wClf
                    ,'--attention_size',att_size, '--num_heads',num_heads
                    ,'--include_item_features','1','--include_context_features','1','--include_immersion','1'
                    ,'--include_source_domain','1','--pretrained','1','--fixed_no','1','--DANN','1'
                    ,'--eval_batch_size','8'
                    # ,'--gpu','2'
                        ] 
                    # subprocess.run(commend_list)
                    content = ' '.join(commend_list)  
                    script_content += content 
                    script_content += '\n'
    return script_content

script_content = ''
script_content = run_AutoInt_DA(script_content)


with open('run_Kuai_AutoInt_ImmersRec.sh','w') as f:
    f.write(script_content)