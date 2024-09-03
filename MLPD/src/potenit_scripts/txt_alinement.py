import os


def main(file_path):
    reconstuct_file(file_path)

def reconstuct_file(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    lines_buffer = []
    for line in lines:
        line_ = line.split('/n')
        import pdb;pdb.set_trace()
        for l in line_:
            lines_buffer.append(f'{l}\n')
    file.close()

    changed_path = os.path.join('.', *path.split('/')[-3:-1], '_eval_results.txt')
    with open(changed_path, 'w') as file:
        file.writelines(lines_buffer)

if __name__ == '__main__':
    main(file_path = '/home/hscho/workspace/src/MLPD/src/jobs/2024-07-17_14h36m_/eval_results.txt')