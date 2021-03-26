import matplotlib.pyplot as plt


def plot_result(original, GT, pred, case_num, path='results/'):
    title = "Case"+str(case_num)
    x = list(range(len(original)))

    figure, axes = plt.subplots(figsize=(15, 6))
    axes.xaxis_date()

    axes.plot(x, original, color = 'red', label = 'Original')
    axes.plot([len(original)], pred, 'ro', color = 'blue', label = 'Predicted BTC Price')
    axes.plot([len(original)], GT, 'ro', color = 'green', label = 'GT BTC Price')
    #axes.xticks(np.arange(0,394,50))
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('BTC Price')
    plt.legend()
    plt.savefig(path+title+'.png')