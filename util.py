def log(file_path,word):
    with open(file_path,'a') as f:
        word+='\n'
        f.write(word)