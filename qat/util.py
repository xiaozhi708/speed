def log(file_path,record):
    with open(file_path,'a') as f:
        record+='\n'
        f.write(record)

if __name__=="__main__":
    log('./a.txt','llll for test')