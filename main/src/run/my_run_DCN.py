import subprocess

# commend_list=['python','main.py', '--model_name', 'DCN', '--dataset', 'KuaiRand_history_test'
#               ,'--include_item_features','1'
#               ,'--include_context_features','1'
#               ,'--include_immersion','1'
#               ,'--include_source_domain','1'
#               ,'--pretrained','1'
#               ,'--fixed_no','1'
#               ,'--DANN','1'
#               ,'--gpu','2'] 
# subprocess.run(commend_list)

def run_DCN_ImmersRec(script_content):
    wDA_list = ['0','0.1','0.01','0.001']
    wClf_list = ['0','0.1','0.01','0.001']
    cross_num_list = ['1','3','6']
    layer_list= ['\'[64]\'','\'[128]\'','\'[64,64]\'']
    for wDA in wDA_list:
        for wClf in wClf_list:
            for cross_num in cross_num_list:
                for layer in layer_list:
                    commend_list=['python','main.py', '--model_name', 'DCN', '--dataset', 'KuaiRand_history'
                        ,'--tradeoff_DA', wDA,'--tradeoff_Clf', wClf
                        ,'--cross_layer_num',cross_num, '--layers',layer
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
script_content = run_DCN_ImmersRec(script_content)


with open('run_Kuai_DCN_ImmersRec.sh','w') as f:
    f.write(script_content)
