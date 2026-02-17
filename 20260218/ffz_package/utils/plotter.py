import matplotlib.pyplot as plt

def auto_plot(data, type='histogram'):
    if type == 'histogram':
        plt.hist(data)
    plt.savefig('plot.png')
    # LaTeX tables
    from pandas import DataFrame
    df = DataFrame(data)
    print(df.to_latex())
