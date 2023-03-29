import torch
from collections import defaultdict

def combine_ne(entities, ne_dict, max_ne_id):
    """
    Combine the adjacent entity in to one, and replace them with placeholder, in ne_dict range
    e.g. B-PERSON I-PERSON A B C => PERSON-0 A B C

    Returns the combined results and the alignment
    """
    assert entities[-1] == ne_dict.eos()
    assert isinstance(entities, torch.Tensor)

    last_ne_start = -1
    combined_ne_tokens = []
    alignment = []
    ne_type_count = defaultdict(int)

    def push_token(current_pose):

        token_to_push = last_ne_token().item()

        last_ne_text = ne_dict[token_to_push]
        assert last_ne_text.startswith('B-') and len(last_ne_text) > 2, last_ne_text

        entity_count = ne_type_count[token_to_push]
        if entity_count + 1 < max_ne_id:
            ne_type_count[token_to_push] += 1

        # the order is B-X, I-X, X-0, X-1. So add 2 here
        combined_id = token_to_push + entity_count + 2
        combined_ne_tokens.append(torch.tensor(combined_id))
        alignment.append(slice(last_ne_start, current_pose))

    def last_ne_token():
        """
        return B-XXX
        """
        assert last_ne_start != -1
        last_ne_token_text = ne_dict[entities[last_ne_start]]
        if last_ne_token_text.startswith('B-'):
            return entities[last_ne_start]
        elif last_ne_token_text.startswith('I-'):
            # In some cases, there will be no "B-XX" and the entity starts with "I-XX"
            return entities[last_ne_start] - 1
        else:
            raise Exception(f'unexpected at {last_ne_start}, {last_ne_token_text}')

    for i, ne_token in enumerate(entities):
        ne_text = ne_dict[ne_token]

        if ne_token == ne_dict.eos():
            assert ne_token == ne_dict.eos()
            if last_ne_start != -1:
                push_token(i)
                last_ne_start = -1
            combined_ne_tokens.append(torch.tensor(ne_dict.eos()))
            alignment.append(slice(i, i+1))
            break
        else:
            if ne_text.startswith('B-'):
                if last_ne_start != -1:
                    push_token(i)
                last_ne_start = i
            elif ne_text.startswith('I-'):
                if last_ne_start == -1 or ne_token != last_ne_token() + 1:
                    # In some cases, there will be no "B-XX" and the entity starts with "I-XX"
                    if last_ne_start != -1:
                        push_token(i)
                    last_ne_start = i
                else:
                    # do nothing, as it's a continuation of entity
                    pass
            elif ne_text == 'O':
                if last_ne_start != -1:
                    push_token(i)
                    last_ne_start = -1
                combined_ne_tokens.append(ne_token)
                alignment.append(slice(i, i+1))
            else:
                raise Exception(f'unexpeced at {i}, token {ne_token}, text {ne_text}')
    if last_ne_start != -1:
        # this means it not ends with EOS. Anyway, put the entities
        push_token(i)
    return combined_ne_tokens, alignment

def combine_ne_with_text(tokens, entities, dictionary, max_ne_id):
    """
    When combine ne, replace the text seq with combined entity
    """
    assert len(tokens) == len(entities)
    combined_ne, alignment = combine_ne(entities, dictionary.ne_dict, max_ne_id)

    combined_tokens = []

    for i, (ne, align) in enumerate(zip(combined_ne, alignment)):
        if ne == dictionary.ne_dict.eos() or dictionary.ne_dict[ne] == 'O':
            combined_tokens.extend(tokens[align])
        else:
            combined_tokens.append(ne + len(dictionary.lang_dict))

    return combined_tokens, combined_ne, alignment

def extract_ne_from_text(tokens, entities, ne_dict, need_type=False, return_pos=False):
    """
    input: [a b c d], [O, B-X, I-X, d]
    output: [(b,c)], [X,]
    """
    # Copy all from CUDA first, or it will be slow
    if type(tokens) != list:
        tokens = tokens.tolist()

    if type(entities) != list:
        entities = entities.tolist()

    assert len(tokens) == len(entities), f'{tokens}, ###, {entities}'

    result = []
    result_type = []
    result_pos = []

    cur_tokens = []
    cur_pos = []
    cur_type = ''

    for i, ne_token in enumerate(entities):
        ne_text = ne_dict[ne_token]

        if ne_token == ne_dict.eos() or ne_token == ne_dict.pad() or ne_text == 'O':
            if cur_tokens:
                result.append(tuple(cur_tokens))
                result_type.append(cur_type)
                result_pos.append((cur_pos[0], cur_pos[-1]))
                
                cur_tokens = []
                cur_type = ''
                cur_pose = []
        elif ne_text.startswith('B-'):
                if cur_tokens:
                    result.append(tuple(cur_tokens))
                    result_type.append(cur_type)
                    result_pos.append((cur_pos[0], cur_pos[-1]))

                cur_tokens = [tokens[i]]
                cur_type = ne_text[2:]
                cur_pos = [i]
        elif ne_text.startswith('I-'):
            if ne_text[2:] == cur_type:
                cur_tokens.append(tokens[i])
                cur_pos.append(i)
            else:
                if cur_tokens:
                    result.append(tuple(cur_tokens))
                    result_type.append(cur_type)
                    result_pos.append((cur_pos[0], cur_pos[-1]))

                cur_tokens = [tokens[i]]
                cur_type = ne_text[2:]
                cur_pos = [i]
        else:
            raise Exception(f'unexpeced at {i}, token {ne_token}, text {ne_text}')

    if cur_tokens:
        # this means it not ends with EOS. Anyway, put the entities
        result.append(tuple(cur_tokens))
        result_type.append(cur_type)
        result_pos.append((cur_pos[0], cur_pos[-1]))
    
    ret = result_pos if return_pos else result

    if need_type:
        return ret, result_type
    else:
        return ret

def tag_entity(tokens, entities, dict):
    entity_pos, entity_type = extract_ne_from_text(tokens, entities, dict.ne_dict, need_type=True, return_pos=True)

    result = tokens.tolist()
    for i in range(len(entity_pos))[::-1]:
        pos = entity_pos[i]
        # use B-XX, I-XX for <XX> </XX>
        b_id = dict.ne_dict.index(f'B-{entity_type[i]}') + len(dict.lang_dict)
        e_id = dict.ne_dict.index(F'I-{entity_type[i]}') + len(dict.lang_dict)

        result = result[:pos[0]] + [b_id] + result[pos[0]: pos[1] + 1] + [e_id] + result[pos[1] + 1:]

    return tokens.new(result)

    