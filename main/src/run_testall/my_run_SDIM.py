
def SDIM_command(data='KR'):
    script_content=''
    for emb_size in ['32','64','128']:
        for dnn_layer in ['\'[64]\'','\'[128]\'','\'[64,64]\'']:
            for att_layer in ['\'[64]\'','\'[64,64]\'']:
                for field in['\'["item","context"]\'','\'["item"]\'']:
                    for _ in range(5):  
                        lr = str(10 ** random.uniform(-4, -2))   
                        l2 = str(random.choice([0, 10**(-4), 10**(-3)]))
                        BPR1 = str(random.choice([0,10**(-4), 10**(-3), 10**(-2)]))
                        BPR2 = str(random.choice([0,10**(-4), 10**(-3), 10**(-2)]))
                        command_list=['python','main.py', '--model_name', 'DIN', '--dataset',data, 
                            '--lr',lr, '--l2', l2,'--emb_size',emb_size, '--layers', layer, '--BPRl2_user', BPR1, '--BPRl2_item', BPR2,
                            '--reuse_hash', '0', '--num_hashes', '1', '--use_sigmoid', '1' ,
                            '--short_target_field', field,'--short_sequence_field',field,'--long_target_field',field,'--long_sequence_field',field,
                            '--att_layers',att_layer, '--dnn_layers', dnn_layer, '--history_max', '20',
                            '--include_item_features','1','--include_context_features','1',
                            '--test_all', '1', '--eval_batch_size','8'] 
                        content = ' '.join(command_list)   
                        script_content += content 
                        script_content += '\n'
    with open(f'command_testall/run_{data}_SDIM_baseline.sh','w') as f:
        f.write(script_content)
    
    script_content=''
    for emb_size in ['32','64','128']:
        for dnn_layer in ['\'[64]\'','\'[128]\'','\'[64,64]\'']:
            for att_layer in ['\'[64]\'','\'[64,64]\'']:
                for field in['\'["item","context"]\'','\'["item"]\'']:
                    for _ in range(5):  
                        lr = str(10 ** random.uniform(-4, -2))   
                        l2 = str(random.choice([0, 10**(-4), 10**(-3)]))
                        BPR1 = str(random.choice([0,10**(-4), 10**(-3), 10**(-2)]))
                        BPR2 = str(random.choice([0,10**(-4), 10**(-3), 10**(-2)]))
                        for _ in range(5):
                            wClf = str(random.choice([0, 10**(-3), 10**(-2),10**(-1)]))
                            wDA = str(random.choice([0, 10**(-3), 10**(-2),10**(-1)]))
                            command_list=['python','main.py', '--model_name', 'DIN', '--dataset', data, 
                                    '--lr',lr, '--l2', l2,'--emb_size',emb_size, '--layers', layer, '--BPRl2_user', BPR1, '--BPRl2_item', BPR2,
                                    '--reuse_hash', '0', '--num_hashes', '1', '--use_sigmoid', '1' ,
                                    '--short_target_field', field,'--short_sequence_field',field,'--long_target_field',field,'--long_sequence_field',field,
                                    '--att_layers',att_layer, '--dnn_layers', dnn_layer, '--history_max', '20',
                                    '--include_item_features','1','--include_context_features','1',
                                    '--tradeoff_DA', wDA,'--tradeoff_Clf', wClf,'--include_immersion','1','--include_source_domain','1','--pretrained','1','--fixed_no','1','--DANN','1',
                                    '--test_all', '1', '--eval_batch_size','8'] 
                            content = ' '.join(command_list)  
                            script_content += content 
                            script_content += '\n'
    with open(f'command_testall/run_{data}_SDIM_ImmersRec.sh','w') as f:
        f.write(script_content)
SDIM_command('KuaiRand')
SDIM_command('MicroVideo')