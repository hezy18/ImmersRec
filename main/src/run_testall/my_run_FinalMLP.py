import subprocess

def FinalMLP_command(data='KuaRand'):
    script_content=''
    for emb_size in ['32','64','128']:
        for layer in ['\'[64]\'','\'[128]\'','\'[64,64]\'']:
            for fs1 in ['\'["user_id","item_id"]\'', '\'["user_id"]\'', '\'[]\'']: 
                for fs2 in ['\'["c_session_order"]\'', '\'[]\'']: 
                    for _ in range(5):  
                        num_head = str(random.uniform(1, 6))   
                        lr = str(10 ** random.uniform(-4, -2))   
                        l2 = str(random.choice([0, 10**(-4), 10**(-3)]))
                        BPR1 = str(random.choice([0,10**(-4), 10**(-3), 10**(-2)]))
                        BPR2 = str(random.choice([0,10**(-4), 10**(-3), 10**(-2)]))
                        command_list=['python','main.py', '--model_name', 'FinalMLP', '--dataset', dataset[data], 
                            '--lr',lr, '--l2', l2,'--emb_size',emb_size, '--layers', layer, '--BPRl2_user', BPR1, '--BPRl2_item', BPR2,
                            '--num_heads',num_heads, '--fs1_context',fs1, '--fs2_context',fs2,
                            '--include_item_features','1','--include_context_features','1',
                            '--test_all', '1', '--eval_batch_size','8'] 
                        content = ' '.join(command_list)   
                        script_content += content 
                        script_content += '\n'
    with open(f'command_testall/run_{data}_FinalMLP_baseline.sh','w') as f:
        f.write(script_content)
    
    script_content=''
    for emb_size in ['32','64','128']:
        for layer in ['\'[64]\'','\'[128]\'','\'[64,64]\'']:
            for fs1 in ['\'["user_id","item_id"]\'', '\'["user_id"]\'', '\'[]\'']: 
                for fs2 in ['\'["c_session_order"]\'', '\'[]\'']: 
                    for _ in range(5):  
                        num_head = str(random.uniform(1, 6))   
                        lr = str(10 ** random.uniform(-4, -2))   
                        l2 = str(random.choice([0, 10**(-4), 10**(-3)]))
                        BPR1 = str(random.choice([0,10**(-4), 10**(-3), 10**(-2)]))
                        BPR2 = str(random.choice([0,10**(-4), 10**(-3), 10**(-2)]))
                        for _ in range(5):
                            wClf = str(random.choice([0, 10**(-3), 10**(-2),10**(-1)]))
                            wDA = str(random.choice([0, 10**(-3), 10**(-2),10**(-1)]))
                            command_list=['python','main.py', '--model_name', 'FinalMLP2', '--dataset', dataset[data], 
                                    '--lr',lr, '--l2', l2,'--emb_size',emb_size, '--layers', layer, '--BPRl2_user', BPR1, '--BPRl2_item', BPR2,
                                    '--num_heads',num_heads, '--fs1_context',fs1, '--fs2_context',fs2,
                                    '--include_item_features','1','--include_context_features','1',
                                    '--tradeoff_DA', wDA,'--tradeoff_Clf', wClf,'--include_immersion','1','--include_source_domain','1','--pretrained','1','--fixed_no','1','--DANN','1',
                                    '--test_all', '1', '--eval_batch_size','8'] 
                            content = ' '.join(command_list)  
                            script_content += content 
                            script_content += '\n'
    with open(f'command_testall/run_{data}_FinalMLP_ImmersRec_simple.sh','w') as f:
        f.write(script_content)
        
FinalMLP_command('KuaiRand')
FinalMLP_command('MicroVideo')