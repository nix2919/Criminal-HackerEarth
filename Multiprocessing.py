from multiprocessing import Pool, cpu_count

def sq(x):
    return x**2

if __name__=='__main__':
    pool = Pool(cpu_count())
    result = pool.map(sq, range(10))
    pool.close()
    pool.join()

    print(result)