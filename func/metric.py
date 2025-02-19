import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import MinCovDet
from rouge_score import rouge_scorer
from sentence_transformers import util
import heapq
import textwrap
# from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore

###### 导入ROUGE评估函数计算ROUGE-L指标
rougeEvaluator = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def count_indent(code):
    if type(code) == str: # a single statement
        return len(code) - len(textwrap.dedent(code))
    elif type(code) == list: # a list of statements, i.e., a function body
        for line in code:
            if line.strip() != '':
                return len(line) - len(textwrap.dedent(line))

def getGenerationRange(generated_ids, tokenizer, passed_first_token=False):
    if passed_first_token:
        start_ind = 1
    else:
        start_ind = 0
    
    try:
        end_ind = generated_ids.index(tokenizer.eos_token_id)
    except:
        end_ind = len(generated_ids)
    
    return start_ind, end_ind

def getCleanGenerationRange(tokenized_generated_text, clean_text, tokenizer):
    try:
        tokenized_generated_text = tokenized_generated_text[:tokenized_generated_text.index(tokenizer.eos_token_id) + 1]
    except:
        pass
    tokenized_clean_text = tokenizer(clean_text, add_special_tokens=False)['input_ids']
    # print(tokenized_generated_text)
    # print(len(tokenized_clean_text))
    # print(clean_text)
    clean_text = clean_text.strip()
    # print(repr(clean_text))
    start_ind, end_ind = None, None
    # print(len(tokenized_generated_text))
    for i in range(len(tokenized_generated_text) - len(tokenized_clean_text) + min(len(tokenized_clean_text) // 2, 50)):
        lo = i + 1
        start_token = tokenizer.decode(tokenized_generated_text[i:i+1])
        if start_token.strip() == '':
            continue
        hi = len(tokenized_generated_text)
        if start_ind is not None:
            break
        while lo <= hi:
            mid = (lo + hi) // 2
            sub_generated_text = tokenizer.decode(tokenized_generated_text[i:mid], skip_special_tokens=True)
            sub_generated_text = sub_generated_text.strip()
            # print(i, lo, hi)
            # if i < 10:
                # print(i, sub_generated_text)
            if sub_generated_text == clean_text:
                end_token = tokenizer.decode(tokenized_generated_text[mid - 1:mid])
                if end_token.strip() == '':
                    hi = mid - 1
                else:
                    start_ind = i
                    end_ind = mid
                    break
            else:
                if len(sub_generated_text) < len(clean_text):
                    lo = mid + 1
                else:
                    hi = mid - 1            
    
    # tokenized_clean_text = tokenizer(clean_text, add_special_tokens=False)['input_ids']
    # print(tokenized_clean_text)
    # start_ind = -1
    # for i in range(len(tokenized_generated_text) - len(tokenized_clean_text) + 1):
    #     if tokenized_generated_text[i:i+len(tokenized_clean_text)] == tokenized_clean_text:
    #         start_ind = i
    #         break
    
    # if start_ind == -1:
    #     return None, None  # Clean text not found in generated text
        
    # end_ind = start_ind + len(tokenized_clean_text)
    
    return start_ind, end_ind

def find_function_body(code, function_name, parser):
    # if type(code) == str:
    #     code = bytes(code, "utf8")
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node

    def search_function(node):
        # 查找类型为function_definition且名称匹配的节点
        for child in node.children:
            if child.type == 'function_definition':
                # 获取函数名
                name_node = child.child_by_field_name('name')
                if name_node and name_node.text.decode('utf8') == function_name:
                    # 找到函数，返回其body部分
                    body_node = child.child_by_field_name('body')
                    return body_node
            # 递归搜索所有子节点
            result = search_function(child)
            if result:
                return result
        return None

    # def search_import(node):
    #     """
    #     Search all import or from_import nodes from a root node.
    #     """
    #     import_nodes = []
    #     for child in node.children:
    #         if child.type == 'import_statement' or child.type == 'import_from_statement':
    #             import_nodes.append(child)
    #         result = search_import(child)
    #         if result:
    #             import_nodes.extend(result)
    #     return import_nodes

    function_body = search_function(root_node)
    # import_nodes = search_import(root_node)
    if function_body:
        if len(function_body.children) == 0: # empty body
            return None
        first_node = function_body.children[0]
        if first_node.type == 'expression_statement' and first_node.children[0].type == 'string':
            start_idx = first_node.end_point[0] + 1
        else:
            start_idx = function_body.start_point[0]
        end_idx = function_body.end_point[0] + 1

        code_lines = code.split("\n")
        body_code = code_lines[start_idx:end_idx]
        if len(body_code) == 0: # empty body
            return None
        # if len(import_nodes) > 0:
        #     body_indent = count_indent(body_code)
        #     for node in import_nodes:
        #         _import = ' '*body_indent + '\n'.join(code_lines[node.start_point[0]:(node.end_point[0] + 1)])
        #         body_code = [_import] + body_code
        #         # print(_import)
        body_code = "\n".join(body_code)
        return body_code
        # return start_idx, end_idx
    else: # no function found
        return None

def getBodyRange(tokenized_generated_text, clean_text, tokenizer, parser, function_name):
    if not f'def {function_name}(' in clean_text:
        completion_lines = clean_text.strip('\n').split("\n")
        body_indent = count_indent(completion_lines[0])
        new_complation = []
        for line in completion_lines:
            if count_indent(line) >= body_indent or line.strip() == "":
                new_complation.append(line)
            else:
                break
        new_complation = "\n".join(new_complation)
        return getCleanGenerationRange(tokenized_generated_text, new_complation, tokenizer)
    else:
        func_body_clean_text = find_function_body(clean_text, function_name, parser)
        if func_body_clean_text is None:
            return getCleanGenerationRange(tokenized_generated_text, clean_text, tokenizer)
        return getCleanGenerationRange(tokenized_generated_text, func_body_clean_text, tokenizer)

def getLineGenerationTokens(tokenized_generated_text, clean_text, tokenizer, parser, function_name):
    if function_name is None:
        start_ind, end_ind = getCleanGenerationRange(tokenized_generated_text, clean_text, tokenizer)
    else:
        start_ind, end_ind = getBodyRange(tokenized_generated_text, clean_text, tokenizer, parser, function_name)
    if start_ind is None or end_ind is None:
        print(f"Cant extract function body token range in {function_name}")
        start_ind = 0
        end_ind = len(tokenized_generated_text)
    last_line_tokens = []
    fl_token = start_ind
    for i in range(start_ind, end_ind):
        if '\n' in tokenizer.decode(tokenized_generated_text[i:i+1]):
            decoded_line = tokenizer.decode(tokenized_generated_text[fl_token:i]).strip()
            if decoded_line != "" and not decoded_line.startswith('#'):
                last_line_tokens.append(i)
            fl_token = i
    if end_ind > 0 and (end_ind - 1) not in last_line_tokens:
        last_line_tokens.append(end_ind - 1)
    return last_line_tokens

def get_function_name(question: str, lang: str):
    func_lines = [x for x in question.strip().split('\n') if x.strip()]

    if lang.lower() == 'python':
        try:
            func_idx = [i for i in range(len(func_lines)) if func_lines[i].startswith("def ")][-1]
            func_name = func_lines[func_idx].split('(')[0].strip()
            # func_prefix = "\n".join(func_lines[:func_idx])
            func_name = func_name[4:].strip()
            return func_name
        except:
            return None
    
    func_name = func_lines[-1].split('{')[0].strip()
    func_prefix = "\n".join(func_lines[:-1])
    return func_name

### 根据GT答案及生成回答计算回答的Rouge Score
def getRouge(rouge, generations, answers):
    # results = rouge.compute(predictions=[generations], references=[answers], use_aggregator=False)
    results = rouge.score(target = answers, prediction = generations)
    RoughL = results["rougeL"].fmeasure  #fmeasure/recall/precision
    return RoughL


def getSentenceSimilarity(generations, answers, SenSimModel):
    gen_embeddings = SenSimModel.encode(generations)
    ans_embeddings = SenSimModel.encode(answers)
    similarity = util.cos_sim(gen_embeddings, ans_embeddings)
    return similarity.item()


### 根据输出logits计算困惑度 scores: ([[logits]],[[logits]],[[logits]])
### logit维度为vocab_size=32000, len(scores)为输出token个数
def get_perplexity_score(scores):
    perplexity = 0.0
    for logits in scores:
        conf = torch.max(logits.softmax(1)).cpu().item()
        perplexity += np.log(conf)
    perplexity = -1.0 * perplexity/len(scores)
    return perplexity



#### 计算token-level energy得分作为幻觉程度度量, scores: ([[logits]],[[logits]],[[logits]])
### logit维度为vocab_size=32000, len(scores)为输出token个数
def get_energy_score(scores):
    avg_energy = 0.0
    for logits in scores:
        # logits[logits<-100]=-100
        # logits[logits>100]=100    ##logit过大会溢出
        # logits = logits/100
        # energy = -torch.log((torch.exp(logits)).sum()).item()
        energy = - torch.logsumexp(logits[0], dim=0, keepdim=False).item()
        avg_energy += energy
    avg_energy = avg_energy/len(scores)
    return avg_energy



#### 根据多次输出计算输出不同sentence的predictive entropy
### batch_scores ([[logits]], [[logits]], [[logits]])
### num_tokens : list 
def get_entropy_score(batch_scores, num_tokens):  
    Conf = []
    for logits in batch_scores: ### logits的维度为num_seq x vocab_size
        conf, index = torch.max(logits.softmax(1), dim=1)
        Conf.append(conf.cpu().numpy())
    Conf = np.array(Conf)  ### Conf的维度为num_tokens x num_seq
    Conf = Conf + 1e-6
    entropy = -1.0 * np.sum(np.log(Conf))/logits.shape[0]
    return entropy



#### 根据多次输出计算输出不同sentence的predictive entropy
### batch_scores (array(logits), array(logits)), 长度为输出句子个数
### logits的维度为num_seq x vocab_size
### num_tokens list类型，每个seq token的个数
def get_lenghthNormalized_entropy(batch_scores, num_tokens):  
    seq_entropy = np.zeros(len(num_tokens))  ## 保存每个seq的log(p)乘积
    for ind1, logits in enumerate(batch_scores): 
        for ind2, seq_logits in enumerate(logits):
            if ind1 < num_tokens[ind2]:
                conf, _ = torch.max(seq_logits.softmax(0), dim=0)
                seq_entropy[ind2] = seq_entropy[ind2] + np.log(conf.cpu().numpy())
    normalized_entropy = 0
    for ind, entropy in enumerate(seq_entropy):
        normalized_entropy += entropy/num_tokens[ind]
    normalized_entropy = -1.0* normalized_entropy/len(num_tokens)
    return normalized_entropy



########## 根据输出的多个回复计算Lexical Similarity
###### generated_texts 为list类型, 输出的文本字符串, 维度为num_seq
def getLexicalSim(generated_texts):
    LexicalSim = 0
    for i in range(len(generated_texts)):
        for j in range(len(generated_texts)):
            if j<=i:
                continue
            LexicalSim += getRouge(rougeEvaluator, generated_texts[i], generated_texts[j])
    LexicalSim = LexicalSim/(len(generated_texts)*(len(generated_texts)-1)/2)
    return LexicalSim


def get_sent_scores_bertscore(best_generation, batch_generations):
    selfcheck_bertscore = SelfCheckBERTScore(rescale_with_baseline=True)
    sent_scores_bertscore = selfcheck_bertscore.predict(
    sentences = best_generation, sampled_passages = batch_generations)
    return sent_scores_bertscore


########## selfcheckGPT复现
###### generated_texts 为list类型, 输出的文本字符串, 维度为num_seq
def getAvgBertScore(bertscore, best_generated_text, generated_texts):
    sent_scores_bertscore = 0
    for item in generated_texts:
        # sent_scores_bertscore += bertscore([item], [best_generated_text])["f1"]
        sent_scores_bertscore += 0  # 速度特别慢, 不用测试selfCheckGPT时关闭该函数
    sent_scores_bertscore = 1 - sent_scores_bertscore/len(generated_texts)
    return sent_scores_bertscore#.cpu().item()


######### 利用nli-roberta-large提出输出语言embedding后计算特征值
def getEigenIndicatorOutput(generated_texts, SenSimModel):
    alpha = 1e-3
    _embeddings = []
    for ind in range(len(generated_texts)):
        embeddings = SenSimModel.encode(generated_texts[ind])
        _embeddings.append(embeddings)
    _embeddings = np.array(_embeddings)
    CovMatrix = np.cov(_embeddings)
    CovMatrix = CovMatrix + alpha*np.eye(CovMatrix.shape[0])
    u, s, vT = np.linalg.svd(CovMatrix)
    eigenIndicatorOutput = np.mean(np.log10(s))
    return eigenIndicatorOutput, s


######### 利用nli-roberta-large提出输出语言embedding后计算特征值
def getEigenScoreOutput(generated_texts, SenSimModel):
    alpha = 1e-3
    _embeddings = []
    for ind in range(len(generated_texts)):
        embeddings = SenSimModel.encode(generated_texts[ind])
        _embeddings.append(embeddings)
    _embeddings = np.array(_embeddings)
    CovMatrix = np.cov(_embeddings)
    CovMatrix = CovMatrix + alpha*np.eye(CovMatrix.shape[0])
    u, s, vT = np.linalg.svd(CovMatrix)
    eigenIndicatorOutput = np.mean(np.log10(s))
    return eigenIndicatorOutput, s


####### 计算协方差矩阵的行列式, 特征采用所有token的均值
####### (num_tokens, num_layers, num_seq, num_input_tokens/1, embedding_size)
def getEigenIndicator(hidden_states): #[num_tokens, 41, num_seq, [n/1], 5120]
    alpha = 1e-3
    selected_layer = int(len(hidden_states[0])/2)
    last_embeddings = torch.zeros(hidden_states[1][-1].shape[0], hidden_states[0][-1].shape[2]).to("cuda")
    for hidden_state in hidden_states[1:]:
        _last_embeddings = hidden_state[selected_layer][:,0,:]
        last_embeddings += _last_embeddings
    last_embeddings/=(len(hidden_states)-1)
    # print(last_embeddings.shape)
    last_embeddings = torch.squeeze(last_embeddings)
    CovMatrix = torch.cov(last_embeddings).cpu().numpy().astype(np.float64)
    u, s, vT = np.linalg.svd(CovMatrix+1.0*np.eye(CovMatrix.shape[0]))
    # eigenIndicator = np.log10(np.prod(s))
    eigenIndicator = np.log10(np.linalg.det(CovMatrix+alpha*np.eye(CovMatrix.shape[0])))
    return eigenIndicator, s



###### 计算最后一个token特征的语义散度的作为句子的语义散度
###### 需要考虑每个句子的长度不一致，去除padding的token的影响
###### hidden_states : (num_tokens, num_layers, num_seq, num_input_tokens/1, embedding_size)
def getEigenIndicator_v0(hidden_states, num_tokens): 
    alpha = 1e-3
    selected_layer = int(len(hidden_states[0])/2)
    # selected_layer = -1
    if len(hidden_states)<2:
        return 0, "None"
    last_embeddings = torch.zeros(hidden_states[1][-1].shape[0], hidden_states[1][-1].shape[2]).to("cuda")
    for ind in range(hidden_states[1][-1].shape[0]):
        last_embeddings[ind,:] = hidden_states[num_tokens[ind]-2][selected_layer][ind,0,:]
    CovMatrix = torch.cov(last_embeddings).cpu().numpy().astype(np.float)
    u, s, vT = np.linalg.svd(CovMatrix+alpha*np.eye(CovMatrix.shape[0]))
    eigenIndicator = np.mean(np.log10(s))
    return eigenIndicator, s

def getMiddleLayerEmbeddingEachToken(hidden_states, num_tokens):
    selected_layer = int(len(hidden_states[0])/2)
    last_embeddings = {}
    for ind in range(hidden_states[1][-1].shape[0]):
        last_embeddings[ind] = []
        for ind1 in range(num_tokens[ind]):
            # last_embeddings[ind].append(hidden_states[ind1][selected_layer][ind,-1,:].detach().cpu().float().numpy().tolist())
            last_embeddings[ind].append(hidden_states[ind1][selected_layer][ind,-1,:].detach().cpu().float().numpy())
        # last_embeddings[ind] = hidden_states[:num_tokens[ind]][selected_layer][ind,:,:]
        # last_embeddings[ind] = [last_embeddings[i][ind,-1,:].detach().cpu().numpy().tolist() for i in range(num_tokens[ind])]
    # print(last_embeddings[0][2])
    # print(len(last_embeddings[0][2]))
    return last_embeddings

def getLayerEmbeddingEachToken(hidden_states, num_tokens, selected_layer):
    last_embeddings = {}
    for ind in range(hidden_states[1][-1].shape[0]):
        last_embeddings[ind] = []
        for ind1 in range(num_tokens[ind]):
            # last_embeddings[ind].append(hidden_states[ind1][selected_layer][ind,-1,:].detach().cpu().float().numpy().tolist())
            last_embeddings[ind].append(hidden_states[ind1][selected_layer][ind,-1,:].detach().cpu().float().numpy())
        # last_embeddings[ind] = hidden_states[:num_tokens[ind]][selected_layer][ind,:,:]
        # last_embeddings[ind] = [last_embeddings[i][ind,-1,:].detach().cpu().numpy().tolist() for i in range(num_tokens[ind])]
    # print(last_embeddings[0][2])
    # print(len(last_embeddings[0][2]))
    return last_embeddings

###### 通过SVD分解计算特征值，从而通过特征值的乘积计算行列式
##### 利用所有token的平均特征作为句子的特征
##### 需要考虑每个句子的长度不一致，去除padding的token的影响
##### hidden_states : (num_tokens, num_layers, num_seq, num_input_tokens/1, embedding_size)
# hidden_states [num_tokens, 41, num_seq, ?, 5120]
def getEigenIndicator_v1(hidden_states, num_tokens): 
    alpha = 1e-3
    selected_layer = int(len(hidden_states[0])/2)
    if len(hidden_states)<2:
        return 0, "None"
    last_embeddings = torch.zeros(hidden_states[1][-1].shape[0], hidden_states[1][-1].shape[2]).to("cuda")
    for ind in range(hidden_states[1][-1].shape[0]):
        for ind1 in range(len(hidden_states)-1):
            if ind1 > num_tokens[ind]-1:
                continue
            last_embeddings[ind,:] += hidden_states[ind1+1][selected_layer][ind,0,:]
        last_embeddings[ind,:] = last_embeddings[ind,:]/(num_tokens[ind]-1)
    CovMatrix = torch.cov(last_embeddings).cpu().numpy().astype(np.float)
    u, s, vT = np.linalg.svd(CovMatrix+alpha*np.eye(CovMatrix.shape[0]))
    eigenIndicator = np.mean(np.log10(s))
    return eigenIndicator, s



def getEigenScore(hidden_states, num_tokens): 
    alpha = 1e-3
    selected_layer = int(len(hidden_states[0])/2)
    if len(hidden_states)<2:
        return 0, "None"
    last_embeddings = torch.zeros(hidden_states[1][-1].shape[0], hidden_states[1][-1].shape[2]).to("cuda")
    for ind in range(hidden_states[1][-1].shape[0]):
        for ind1 in range(len(hidden_states)-1):
            if ind1 > num_tokens[ind]-1:
                continue
            last_embeddings[ind,:] += hidden_states[ind1+1][selected_layer][ind,0,:]
        last_embeddings[ind,:] = last_embeddings[ind,:]/(num_tokens[ind]-1)
    CovMatrix = torch.cov(last_embeddings).cpu().numpy().astype(np.float)
    u, s, vT = np.linalg.svd(CovMatrix+alpha*np.eye(CovMatrix.shape[0]))
    eigenIndicator = np.mean(np.log10(s))
    return eigenIndicator, s



###### 查看不同层的语义熵, 利用每句话最后一个token的特征计算语义熵, 融合不同层的语义熵
###### hidden_states : (num_tokens, num_layers, num_seq, num_input_tokens/1, embedding_size)
def getEigenIndicator_v2(hidden_states, num_tokens):
    alpha = 1e-3
    LayerEigens = []
    if len(hidden_states)<2:
        return 0, "None"
    for layer_ind in range(len(hidden_states[0])):
        last_embeddings = torch.zeros(hidden_states[1][-1].shape[0], hidden_states[1][-1].shape[2]).to("cuda")
        for seq_ind in range(hidden_states[1][-1].shape[0]):
            for token_ind in range(len(hidden_states)-1):
                if token_ind > num_tokens[seq_ind]-1:
                    continue
                last_embeddings[seq_ind,:] += hidden_states[token_ind+1][layer_ind][seq_ind,0,:]
            last_embeddings[seq_ind,:] = last_embeddings[seq_ind,:]/(num_tokens[seq_ind]-1)
        CovMatrix = torch.cov(last_embeddings).cpu().numpy().astype(np.float)
        u, s, vT = np.linalg.svd(CovMatrix+alpha*np.eye(CovMatrix.shape[0]))
        eigenIndicator = np.mean(np.log10(s))
        LayerEigens.append(eigenIndicator)
    LayerEigens = np.array(LayerEigens)
    print("LayerEigens: ", LayerEigens)
    return np.mean(LayerEigens[20:-2]), s

def getNumToken(generated_text, cleaned_text, tokenizer):
    tokenized_generated_text = tokenizer.tokenize(generated_text, add_special_tokens=False)
    tokenized_cleaned_text = tokenizer.tokenize(cleaned_text, add_special_tokens=False)
    num_tokens = len(tokenized_generated_text) - len(tokenized_cleaned_text)
    return num_tokens

# ###### 查看不同层的语义熵, 融合不同层的语义熵
# def getEigenIndicator_v2(hidden_states): #[num_tokens, 41, num_seq, ?, 5120]
#     alpha = 1e-3
#     LayerEigens = []
#     if len(hidden_states)<2:
#         return 0, "None"
#     for layer_ind in range(len(hidden_states[0])):
#         last_embeddings = torch.zeros(hidden_states[1][-1].shape[0], hidden_states[1][-1].shape[2]).to("cuda")
#         for hidden_state in hidden_states[1:]:
#             _last_embeddings = hidden_state[layer_ind][:,0,:]
#             last_embeddings += _last_embeddings
#         last_embeddings/=(len(hidden_states)-1)
#         last_embeddings = last_embeddings[:,::20]
#         CovMatrix = torch.cov(last_embeddings).cpu().numpy().astype(np.float)
#         u, s, vT = np.linalg.svd(CovMatrix+alpha*np.eye(CovMatrix.shape[0]))
#         eigenIndicator = np.mean(np.log10(s))
#         LayerEigens.append(eigenIndicator)
#     LayerEigens = np.array(LayerEigens)
#     print("LayerEigens: ", LayerEigens)
#     return np.mean(LayerEigens[10:-2]), s



###### 融合不同层的特征作为语义embedding
def getEigenIndicator_v3(hidden_states): #[num_tokens, 41, num_seq, ?, 5120]
    alpha = 1e-3
    layer_ind_min = 10
    layer_ind_max = 35
    if len(hidden_states)<2:
        return 0, "None"
    last_embeddings = torch.zeros(hidden_states[1][-1].shape[0], hidden_states[1][-1].shape[2]).to("cuda")
    for hidden_state in hidden_states[1:]:
        _last_embeddings = torch.zeros(last_embeddings.shape).to("cuda")
        for k in range(len(hidden_state)):
            if k < layer_ind_min or k > layer_ind_max:
                continue
            _last_embeddings += hidden_state[k][:,0,:]
        last_embeddings += _last_embeddings/(layer_ind_max-layer_ind_min)
    last_embeddings/=(len(hidden_states)-1)
    CovMatrix = torch.cov(last_embeddings).cpu().numpy().astype(np.float)
    u, s, vT = np.linalg.svd(CovMatrix+alpha*np.eye(CovMatrix.shape[0]))
    eigenIndicator = np.mean(np.log10(s))
    return eigenIndicator, s


###### 计算特征维度的协方差矩阵
def getEigenIndicator_v4(hidden_states): #[num_tokens, 41, num_seq, ?, 5120]
    alpha = 1e-3
    if len(hidden_states)<2:
        return 0, "None"
    last_embeddings = torch.zeros(hidden_states[1][-1].shape[0], hidden_states[1][-1].shape[2]).to("cuda")
    for hidden_state in hidden_states[1:]:
        _last_embeddings = hidden_state[-2][:,0,:]
        last_embeddings += _last_embeddings
    last_embeddings/=(len(hidden_states)-1)
    last_embeddings = torch.squeeze(last_embeddings)
    last_embeddings = last_embeddings[:,::40]
    CovMatrix = torch.cov(last_embeddings.transpose(0,1)).cpu().numpy().astype(np.float)
    u, s, vT = np.linalg.svd(CovMatrix+alpha*np.eye(CovMatrix.shape[0]))
    eigenIndicator = np.mean(np.log10(s))
    return eigenIndicator, s



###### 利用深度特征计算似然概率密度
def getEigenIndicator_v5(hidden_states, features): #[num_tokens, 41, num_seq, ?, 5120]
    alpha = 1e-3
    if len(hidden_states)<2:
        return 0, "None"
    last_embeddings = torch.zeros(hidden_states[1][-1].shape[0], hidden_states[1][-1].shape[2]).to("cuda")
    for hidden_state in hidden_states[1:]:
        _last_embeddings = hidden_state[-2][:,0,:]
        last_embeddings += _last_embeddings
    last_embeddings/=(len(hidden_states)-1)
    features = features[::40].cpu().numpy()
    last_embeddings = last_embeddings[:,::40].cpu().numpy()
    # last_embeddings = sample_selected(last_embeddings, features)
    Mean = np.mean(last_embeddings, axis=0)
    CovMatrix = np.cov(last_embeddings.transpose())
    print(CovMatrix)
    CovMatrix = CovMatrix + alpha*np.eye(CovMatrix.shape[0])
    pro = np.matmul(np.matmul((features-Mean).reshape(1,-1), np.linalg.inv(CovMatrix)), (features-Mean).reshape(-1,1))
    # pro = np.exp(-0.5*np.matmul(np.matmul((features-Mean).reshape(1,-1), np.linalg.inv(CovMatrix)), (features-Mean).reshape(-1,1)))
    # pro = pro[0][0]/np.sqrt(np.linalg.det(CovMatrix))
    u, s, vT = np.linalg.svd(CovMatrix)
    pro = -pro[0][0] - np.sum(np.log(s))
    return pro, "None"



######### 提取most_likely_generation的特征embedding
def get_features(hidden_states):
    last_embeddings = torch.zeros(hidden_states[0][-1].shape[-1]).to("cuda")
    for hidden_state in hidden_states[1:]:
        _last_embeddings = hidden_state[-2][0,0,:]
        last_embeddings += _last_embeddings
    last_embeddings/=(len(hidden_states)-1)
    return last_embeddings



def sample_selected(last_embeddings, features):
    dist = []
    for k in range(last_embeddings.shape[0]):
        dist.append(np.linalg.norm(last_embeddings[k,:]-features)+1e-12*np.random.random())
    temp_dist = heapq.nsmallest(int(0.5*last_embeddings.shape[0]), dist)
    index = [dist.index(i) for i in temp_dist]
    last_embeddings = last_embeddings[index,:]
    print(index)
    print(last_embeddings.shape)
    return last_embeddings



def ParameterClip(model):
    # for name, param in model.named_parameters():
        # if name == "lm_head.weight":
        #     np.save("./data/features/lm_head_weight.npy", param.cpu().numpy())
    ratio_high = 0.1
    ratio_low = 0.3
    lm_head_weight = np.load("./data/features/lm_head_weight.npy")
    weight_importance = np.load("./data/features/weight_importance.npy")
    k_high = int(ratio_high/100*weight_importance.shape[1])
    k_low = int(ratio_low/100*weight_importance.shape[1])
    for i in range(weight_importance.shape[0]):
        # value_max, ind_max = torch.topk(torch.tensor(weight_importance[i,:]), k_high)
        value_min, ind_min = torch.topk(torch.tensor(weight_importance[i,:]), k_low, largest=False)
        # weight_importance[i,:][weight_importance[i,:]>= value_max.numpy()[-1]] = 0
        weight_importance[i,:][weight_importance[i,:] <= value_min.numpy()[-1]] = -1000
        weight_importance[i,:][weight_importance[i,:] > value_min.numpy()[-1]] = 1
        weight_importance[i,:][weight_importance[i,:] ==-1000] = 0
    print(weight_importance)
    head_weights_op = weight_importance*lm_head_weight
    model.state_dict()["lm_head.weight"].copy_(torch.tensor(head_weights_op))
    return model



def ParameterClip_v1(model):
    # for name, param in model.named_parameters():
        # if name == "lm_head.weight":
        #     np.save("./data/features/lm_head_weight.npy", param.cpu().numpy())
    ratio_high = 0.0001
    ratio_low = 0.001
    lm_head_weight = np.load("./data/features/lm_head_weight.npy")
    weight_importance = np.load("./data/features/weight_importance1.npy")
    # k_high = int(ratio_high/100*weight_importance.shape[0]*weight_importance.shape[1])
    k_low = int(ratio_low/100*weight_importance.shape[0]*weight_importance.shape[1])
    # value_max, ind_max = torch.topk(torch.tensor(weight_importance.flatten()), k_high)
    value_min, ind_min = torch.topk(torch.tensor(weight_importance.flatten()), k_low, largest=False)
    # weight_importance[weight_importance >= value_max.numpy()[-1]] = 1000
    # weight_importance[weight_importance < value_max.numpy()[-1]] = 1
    # weight_importance[weight_importance == 1000] = 0
    weight_importance[weight_importance <= value_min.numpy()[-1]] = -1000
    weight_importance[weight_importance > value_min.numpy()[-1]] = 1
    weight_importance[weight_importance ==-1000] = 0
    print(weight_importance)
    head_weights_op = weight_importance*lm_head_weight
    model.state_dict()["lm_head.weight"].copy_(torch.tensor(head_weights_op))
    return model



def ParameterClip_v2(model):
    ratio_high = 0.1
    ratio_low = 1
    lm_head_weight = np.load("./data/features/lm_head_weight.npy")
    weight_importance = np.load("./data/features/weight_importance.npy")
    weight_importance = np.linalg.norm(weight_importance, axis=0)
    k_high = int(ratio_high/100*4096)
    k_low = int(ratio_low/100*4096)
    value_max, ind_max = torch.topk(torch.tensor(weight_importance), k_high)
    value_min, ind_min = torch.topk(torch.tensor(weight_importance), k_low, largest=False)
    print(ind_max)
    lm_head_weight[:, ind_min] = 0
    model.state_dict()["lm_head.weight"].copy_(torch.tensor(lm_head_weight))
    return model



# ###### 通过SVD分解计算特征值，从而通过特征值的乘积计算行列式
# def getEigenIndicator_v1(hidden_states): #[num_tokens, 41, num_seq, ?, 5120]
#     alpha = 1e-3
#     if len(hidden_states)<2:
#         return 0, "None"
#     last_embeddings = torch.zeros(hidden_states[1][-1].shape[0], hidden_states[1][-1].shape[2]).to("cuda")
#     for hidden_state in hidden_states[1:]:
#         _last_embeddings = hidden_state[-2][:,0,:]
#         last_embeddings += _last_embeddings
#     last_embeddings/=(len(hidden_states)-1)
#     last_embeddings = torch.squeeze(last_embeddings)
#     CovMatrix = torch.cov(last_embeddings).cpu().numpy().astype(np.float)
#     u, s, vT = np.linalg.svd(CovMatrix+alpha*np.eye(CovMatrix.shape[0]))
#     eigenIndicator = np.mean(np.log10(s))
#     return eigenIndicator, s


# ###### 计算最后一个token特征的语义散度的作为句子的语义散度
# def getEigenIndicator_v6(hidden_states, num_tokens): #[num_tokens, 41, num_seq, ?, 5120]
#     alpha = 1e-3
#     if len(hidden_states)<2:
#         return 0, "None"
#     last_embeddings = torch.zeros(hidden_states[1][-1].shape[0], hidden_states[1][-1].shape[2]).to("cuda")
#     for ind in range(hidden_states[1][-1].shape[0]):
#         last_embeddings[ind,:] = hidden_states[num_tokens[ind]-2][-10][ind,0,:]
#     CovMatrix = torch.cov(last_embeddings).cpu().numpy().astype(np.float)
#     u, s, vT = np.linalg.svd(CovMatrix+alpha*np.eye(CovMatrix.shape[0]))
#     eigenIndicator = np.mean(np.log10(s))
#     return eigenIndicator, s


# ###### 通过特征计算每个token的概率
# def getEigenIndicator_v6(hidden_states, Mean, CovMatrixInv): #[num_tokens, 41, num_seq, ?, 5120]
#     alpha = 1e-3
#     if len(hidden_states)<2:
#         return 0, "None"
#     Dist = np.zeros(len(hidden_states)-1)
#     for i in range(len(hidden_states)):
#         features = hidden_states[i+1][-1][0][0].cpu().numpy()
#         Dist[i] = np.matmul(np.matmul((features-Mean).reshape(1,-1), CovMatrixInv), (features-Mean).reshape(-1,1))
#     return np.mean(Dist)/10000



#### 根据多次输出计算输出不同sentence的predictive entropy 
#### batch_scores为多个sentence的输出scores
# def get_entropy_score_v0(batch_scores):
#     Conf = []
#     for logits in batch_scores:
#         conf, index = torch.max(logits.softmax(1), dim=1)
#         Conf.append(conf.cpu().numpy())
#     Conf = np.array(Conf)
#     sentence_probability = np.prod(Conf, axis=0)
#     sentence_probability += 1e-6  ### 防止概率为0
#     sentence_probability = sentence_probability/np.sum(sentence_probability)
#     entropy = -1*np.sum(sentence_probability*np.log(sentence_probability))
#     return entropy



#### 根据多次输出计算输出不同sentence的predictive entropy
### batch_scores ([[logits]], [[logits]], [[logits]])
# def get_entropy_score_v1(batch_scores):  
#     Conf = []
#     for logits in batch_scores: ### logits的维度为num_seq x vocab_size
#         conf, index = torch.max(logits.softmax(1), dim=1)
#         Conf.append(conf.cpu().numpy())
#     Conf = np.array(Conf)  ### Conf的维度为num_tokens x num_seq
#     Conf = Conf + 1e-6
#     entropy = -1.0 * np.sum(np.log(Conf))/10.0
#     return entropy



# ###### 计算token级别语义散度
# def getEigenIndicator_v3(hidden_states): #[num_tokens, 41, num_seq, ?, 5120]
#     alpha = 1e-3
#     last_embeddings = []
#     for hidden_state in hidden_states[1:]:
#         _last_embeddings = hidden_state[-1][:,0,:]
#         print(_last_embeddings.shape)
#         last_embeddings.append(_last_embeddings)
#     last_embeddings = torch.cat(last_embeddings, dim=1)/(len(hidden_states)-1)
#     CovMatrix = torch.cov(last_embeddings).cpu().numpy().astype(np.float)
#     print(CovMatrix.shape)
#     print(CovMatrix)

#     u, s, vT = np.linalg.svd(CovMatrix+alpha*np.eye(CovMatrix.shape[0]))
#     eigenIndicator = np.log10(np.prod(s))
#     print(s)
#     # eigenIndicator = 1-(np.max(np.log(s))/np.sum(np.log(s)))
#     # eigenIndicator = (np.max(np.log(s)) - np.sort(np.log(s))[-2])/np.max(np.log(s))
#     return eigenIndicator, s



# ###### 通过SVD分解计算特征值, 计算token最后一个/每个token位置的语义熵的和, 没效果
# def getEigenIndicator_v4(hidden_states): #[num_tokens, 41, num_seq, ?, 5120]
#     alpha = 1e-3
#     eigenIndicator = 0.0
#     # for hidden_state in hidden_states[1:]:
#     for hidden_state in hidden_states[1:]:
#         last_embeddings = hidden_state[-1][:,0,:]
#         CovMatrix = torch.cov(last_embeddings).cpu().numpy().astype(np.float)
#         u, s, vT = np.linalg.svd(CovMatrix+alpha*np.eye(CovMatrix.shape[0]))
#         eigenIndicator = eigenIndicator + 1.0*np.log(np.prod(s))
#     eigenIndicator = eigenIndicator/(len(hidden_states)-1)
#     return eigenIndicator, "None"




######  鲁邦的协方差估计(调用scikit-learn函数), 抑制噪声tokens带来的影响
# def getEigenIndicator_v2(hidden_states): #[num_tokens, 41, num_seq, ?, 5120]
#     alpha = 1e-6
#     last_embeddings = torch.zeros(hidden_states[1][-1].shape[0], hidden_states[0][-1].shape[2]).to("cuda")
#     for hidden_state in hidden_states[1:]:
#         _last_embeddings = hidden_state[-1][:,0,:]
#         last_embeddings += _last_embeddings
#     last_embeddings/=(len(hidden_states)-1)
#     last_embeddings = torch.squeeze(last_embeddings)
#     # CovMatrix = torch.cov(last_embeddings).cpu().numpy().astype(np.float)
#     RobustCov = MinCovDet().fit(np.transpose(last_embeddings.cpu().numpy()))
#     CovMatriv = RobustCov.covariance_
#     u, s, vT = np.linalg.svd(CovMatriv+alpha*np.eye(CovMatriv.shape[0]))
#     eigenIndicator = np.log(np.prod(s))
#     # eigenIndicator = 1-(np.max(np.log(s))/np.sum(np.log(s)))
#     # eigenIndicator = (np.max(np.log(s)) - np.sort(np.log(s))[-2])/np.max(np.log(s))
#     return eigenIndicator


# ###### 通过计算piar-wise距离度量语义空间的散度
# def getEigenIndicator_v8(hidden_states): #[num_tokens, 41, num_seq, ?, 5120]
#     alpha = 1e-3
#     dist = 0
#     if len(hidden_states)<2:
#         return 0, "None"
#     last_embeddings = torch.zeros(hidden_states[1][-1].shape[0], hidden_states[1][-1].shape[2]).to("cuda")
#     for hidden_state in hidden_states[1:]:
#         _last_embeddings = hidden_state[-2][:,0,:]
#         last_embeddings += _last_embeddings
#     last_embeddings/=(len(hidden_states)-1)
#     for i in range(last_embeddings.shape[0]):
#         for j in range(last_embeddings.shape[0]):
#             dist = dist+torch.norm(last_embeddings[i,:]-last_embeddings[j,:]).item()
#     return dist, "None"



###### 利用Non-local形式进行特征融合
# def getEigenIndicator_v2(hidden_states): #[num_tokens, 41, num_seq, ?, 5120]
#     alpha = 1e-3
#     last_embeddings = []
#     for hidden_state in hidden_states[1:]:
#         _last_embeddings = hidden_state[-1][:,0,:]
#         last_embeddings.append(_last_embeddings)
#     last_embeddings = torch.stack(last_embeddings, dim=0)  # num_tokens x num_seq x voc_size
#     _NonLocalFea = []
#     for ind in range(last_embeddings.shape[1]):
#         fea = last_embeddings[:,ind,:].cpu().numpy()  # num_tokes x voc_size
#         NonLocalfea = np.matmul(np.matmul(fea, np.transpose(fea))+alpha*np.eye(fea.shape[0]), fea)
#         NonLocalfea = torch.tensor(NonLocalfea).to("cuda")
#         _NonLocalFea.append(NonLocalfea.mean(dim=0))
#     _NonLocalFea = torch.stack(_NonLocalFea, dim=0)
#     print(_NonLocalFea)
#     print(_NonLocalFea.shape)
#     CovMatrix = torch.cov(_NonLocalFea).cpu().numpy().astype(np.float)
#     print(CovMatrix)
#     u, s, vT = np.linalg.svd(CovMatrix+alpha*np.eye(CovMatrix.shape[0]))
#     eigenIndicator = np.log10(np.prod(s))
#     print(s)
#     return eigenIndicator, s
