# Extractive QA
Данная задача рассматривалась как задача Extractive QA. Во время исследования было опробовано несколько различных методов, о которых сейчас подробнее:

*Все исследования проходили с использованием сервиса wandb, где можно было подробно отслеживать изменения метрик с учетом  гиперпараметров и подходов.*
## Гипотезы и предположения
1. Исходя из исследований текстов в зависимости от лейбла (подробнее в ```Data exploration.ipynb```) было предположение, что обучение двух моделей даст для каждого лейбла отдельно даст более хороший результат, что оказалось неверно. 
2. Было предположение, что разбиение одного текста на несколько текстов меньшей длины со stride=N может улучшить метрики из-за возможности использования моделей с бОльшим количеством параметров, чем используемая ```cointegrated/rubert-tiny2```, но с меньшей длиной контекста (512 против 2048 у  rubert-tiny2). Также предположение оказалось неверным и результаты ухудшились.
3. Были попытки обучить модель ```cointegrated/rubert-tiny2``` на датасете ```Sberquad```, чтобы потом дообучить на исходном датасете задачи. Это было затратно в вычислительном плане и не принесло ожидаемых результатов. В данном случае мы делаем в выбор в пользу скорости. 
## Используемые технические особенности при обучении
1. Gradient Clipping.
2. Различные lr_scheduler (linear, cosine, polynomial).
3. Заморозка энкодер модели и обучение только последнего линейного слоя около 3000 шагов, потом размораживание энкодера и дообучение еще несколько эпох. (Не используется в финальном решении)
### Ссылки и Примечания
[Веса финальной модели (gdrive)](https://drive.google.com/drive/folders/1_FNojiT2LJDH3C2slfoFXh-Z8ufdKbP4?usp=sharing)

Все сиды зафиксированы, можно воспроизвести:

```train.py --file_path train.json --test_size 0``` 

