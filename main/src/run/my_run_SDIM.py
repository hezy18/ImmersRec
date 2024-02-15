import subprocess

commend_list=['python','main.py', '--model_name', 'SDIM', '--dataset', 'KuaiRand_history_test'
              ,'--include_item_features','1', '--include_his_context', '1'
              ,'--include_context_features','1'
              ,'--include_immersion','1'
              ,'--include_source_domain','1'
              ,'--pretrained','1'
              ,'--fixed_no','1'
              ,'--DANN','1'
              ,'--gpu','0'
              ,'--regenerate','1'] 
# subprocess.run(commend_list)


def run_SDIM_ImmersRec(script_content, dataset):
    # att_list=['\'[32]\'','\'[64]\'','\'[64,64]\'']
    wDA_list = ['0.1','0.01','0.001']
    wClf_list = ['0.1','0.01','0.001']
    dnn_list=['\'[64]\'','\'[64,64]\'','\'[64,64,64]\'']
    reuse_list=['0','1']
    for reuse in reuse_list:
        for sigmoid in ['0','1']:
            for field in['\'["item","context"]\'']:
                for dnn in dnn_list:
                    for wDA in wDA_list:
                        for wClf in wClf_list:
                            commend_list=['python','main.py', '--model_name', 'SDIM', '--dataset', dataset
                                        ,'--dnn_hidden_units', dnn, '--reuse_hash', reuse, '--num_hashes', '1', '--history_max', '20' 
                                        ,'--use_sigmoid', sigmoid ,'--tradeoff_DA', wDA,'--tradeoff_Clf', wClf, '--include_his_context', '1'
                                        ,'--short_target_field', field,'--short_sequence_field',field,'--long_target_field',field,'--long_sequence_field',field
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
script_content = run_SDIM_ImmersRec(script_content,'KuaiRand_history')

    
with open('run_Kuai_SDIM_ImmersRec.sh','w') as f:
    f.write(script_content)
