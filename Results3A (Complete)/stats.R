library(ggplot2)

results3_20 <- read.csv('Results3A_3_20.csv')
results5_20 <- read.csv('Results3A_5_20.csv')
results7_20 <- read.csv('Results3A_7_20.csv')
epoch <- results3_20$epoch
error3_20 <- results3_20$error
error5_20 <- results5_20$error
error7_20 <- results7_20$error
error20 <- data.frame(epoch, error3_20, error5_20, error7_20)
ggplot(error20) + geom_line(aes(x=epoch, y=error3_20, color="Grado 3")) 
+ geom_line(aes(x=epoch, y=error5_20, color="Grado 5")) 
+ geom_line(aes(x=epoch, y=error7_20, color="Grado 7")) 
+ scale_color_discrete(name="Grado del polinomio") 
+ ylab('Error cuadrático medio') 
+ theme(legend.key.size = unit(0.5, 'cm'), legend.title = element_text(size=10))

results3_50 <- read.csv('Results3A_3_50.csv')
results5_50 <- read.csv('Results3A_5_50.csv')
results7_50 <- read.csv('Results3A_7_50.csv')
epoch <- results3_50$epoch
error3_50 <- results3_50$error
error5_50 <- results5_50$error
error7_50 <- results7_50$error
error50 <- data.frame(epoch, error3_50, error5_50, error7_50)
ggplot(error50) + geom_line(aes(x=epoch, y=error3_50, color="Grado 3")) + geom_line(aes(x=epoch, y=error5_50, color="Grado 5")) + geom_line(aes(x=epoch, y=error7_50, color="Grado 7")) + scale_color_discrete(name="Grado del polinomio") + ylab('Error cuadrático medio') + theme(legend.key.size = unit(0.5, 'cm'), legend.title = element_text(size=10))


results3_100 <- read.csv('Results3A_3_100.csv')
results5_100 <- read.csv('Results3A_5_100.csv')
results7_100 <- read.csv('Results3A_7_100.csv')
epoch <- results3_100$epoch
error3_100 <- results3_100$error
error5_100 <- results5_100$error
error7_100 <- results7_100$error
error100 <- data.frame(epoch, error3_100, error5_100, error7_100)
ggplot(error100) + geom_line(aes(x=epoch, y=error3_100, color="Grado 3")) + geom_line(aes(x=epoch, y=error5_100, color="Grado 5")) + geom_line(aes(x=epoch, y=error7_100, color="Grado 7")) + scale_color_discrete(name="Grado del polinomio") + ylab('Error cuadrático medio') + theme(legend.key.size = unit(0.5, 'cm'), legend.title = element_text(size=10))