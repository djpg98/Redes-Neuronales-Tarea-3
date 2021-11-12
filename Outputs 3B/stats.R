library(ggplot2)

#En Python tomé la primera fila como header accidentalmente,
#en lugar de como dato. Me di cuenta la noche de la entrega, así que tuve que
#hacer lo mismo aquí para efectos de las comparaciones
original <- read.csv('../interpolacionpolinomial_real.csv', header=T)
original_sorted <- original[order(original$X0.25192),]

train <- read.csv('../interpolacionpolinomial_train.csv', header=T)
train_sorted <- train[order(train$X0.25192),]

pol_3_00001 <-read.csv('output_3_00001.csv')
pol_3_0001 <-read.csv('output_3_0001.csv')
pol_3_001 <-read.csv('output_3_001.csv')
pol_3_01 <-read.csv('output_3_01.csv')

pol_5_00001 <-read.csv('output_5_00001.csv')
pol_5_0001 <-read.csv('output_5_0001.csv')
pol_5_001 <-read.csv('output_5_001.csv')

pol_7_00001 <-read.csv('output_7_00001.csv')
pol_7_0001 <-read.csv('output_7_0001.csv')
pol_7_001 <-read.csv('output_7_001.csv')

xoriginal <- original_sorted$X0.25192
yoriginal <- original_sorted$X2.5168
xpol <- pol_3_01$in_val

y3 <- pol_3_00001$out_val
y5 <- pol_5_00001$out_val
y7 <- pol_7_00001$out_val
results00001 <- data.frame(xoriginal, yoriginal, xpol, y3, y5, y7)
ggplot(results00001) + geom_line(aes(x=xoriginal, y=yoriginal, color="Real")) + geom_line(aes(x=xpol, y=y3, color="Grado 3")) + geom_line(aes(x=xpol, y=y5, color="Grado 5")) + geom_line(aes(x=xpol, y=y7, color="Grado 7")) + scale_color_discrete(name="Polinomio") + xlab("Entrada") + ylab("Salida") + theme(legend.key.size = unit(0.5, 'cm'), legend.title = element_text(size=10))

y3 <- pol_3_0001$out_val
y5 <- pol_5_0001$out_val
y7 <- pol_7_0001$out_val
results0001 <- data.frame(xoriginal, yoriginal, xpol, y3, y5, y7)
ggplot(results0001) + geom_line(aes(x=xoriginal, y=yoriginal, color="Real")) + geom_line(aes(x=xpol, y=y3, color="Grado 3")) + geom_line(aes(x=xpol, y=y5, color="Grado 5")) + geom_line(aes(x=xpol, y=y7, color="Grado 7")) + scale_color_discrete(name="Polinomio") + xlab("Entrada") + ylab("Salida") + theme(legend.key.size = unit(0.5, 'cm'), legend.title = element_text(size=10))

y3 <- pol_3_001$out_val
y5 <- pol_5_001$out_val
y7 <- pol_7_001$out_val
results001 <- data.frame(xoriginal, yoriginal, xpol, y3, y5, y7)
ggplot(results001) + geom_line(aes(x=xoriginal, y=yoriginal, color="Real")) + geom_line(aes(x=xpol, y=y3, color="Grado 3")) + geom_line(aes(x=xpol, y=y5, color="Grado 5")) + geom_line(aes(x=xpol, y=y7, color="Grado 7")) + scale_color_discrete(name="Polinomio") + xlab("Entrada") + ylab("Salida") + theme(legend.key.size = unit(0.5, 'cm'), legend.title = element_text(size=10))

y3 <- pol_3_01$out_val
results01 <- data.frame(xoriginal, yoriginal, xpol, y3)
ggplot(results01) + geom_line(aes(x=xoriginal, y=yoriginal, color="Real")) + geom_line(aes(x=xpol, y=y3, color="Grado 3")) + scale_color_discrete(name="Polinomio") + xlab("Entrada") + ylab("Salida") + theme(legend.key.size = unit(0.5, 'cm'), legend.title = element_text(size=10))

y7 <- pol_7_001$out_val
ytrain <- train_sorted$X2.3573
resultsFinal <- data.frame(xoriginal, yoriginal, ytrain, y7)
ggplot(resultsFinal) + geom_line(aes(x=xoriginal, y=yoriginal, color="Real")) + geom_line(aes(x=xpol, y=y3, color="Grado 7")) + geom_point(aes(x=xoriginal, y=ytrain, color="Datos de entrenamiento")) + scale_color_discrete(name="Polinomio") + xlab("Entrada") + ylab("Salida") + theme(legend.key.size = unit(0.5, 'cm'), legend.title = element_text(size=10))
