import subprocess


def run_SDIM_baseline(script_content):
    dnn_list=['\'[64]\'','\'[64,64]\'','\'[64,64,64]\'']
    reuse_list=['0','1']
    for reuse in reuse_list:
        for sigmoid in ['0','1']:
            for field in['\'["item","context"]\'','\'["item"]\'']:
                for dnn in dnn_list:
                    commend_list=['python','main.py', '--model_name', 'SDIM', '--dataset', 'KuaiRand_history'
                    ,'--dnn_hidden_units', dnn, '--reuse_hash', reuse, '--num_hashes', '1', '--history_max', '20' 
                    ,'--include_his_context', '1'
                    ,'--short_target_field', field,'--short_sequence_field',field,'--long_target_field',field,'--long_sequence_field',field
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
script_content = run_SDIM_baseline(script_content)


with open('run_Kuai_SDIM_baseline.sh','w') as f:
    f.write(script_content)