#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${1:-.}"
echo "Downloading Minzdrav clinical recommendations into: $BASE_DIR/docs/01_raw/minzdrav"
mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/cardiology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/cardiology/КР154_4.pdf" ]; then
  echo "Downloading КР154_4.pdf — Острый коронарный синдром без подъема сегмента ST электрокардиограммы"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/cardiology/КР154_4.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=154_4" || echo "FAILED: КР154_4.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=154_4)"
else
  echo "Exists, skipped: КР154_4.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/cardiology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/cardiology/КР159_2.pdf" ]; then
  echo "Downloading КР159_2.pdf — Легочная гипертензия, в том числе хроническая тромбоэмболическая легочная гипертензия"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/cardiology/КР159_2.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=159_2" || echo "FAILED: КР159_2.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=159_2)"
else
  echo "Exists, skipped: КР159_2.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/cardiology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/cardiology/КР160_1.pdf" ]; then
  echo "Downloading КР160_1.pdf — Брадиаритмии и нарушения проводимости"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/cardiology/КР160_1.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=160_1" || echo "FAILED: КР160_1.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=160_1)"
else
  echo "Exists, skipped: КР160_1.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/cardiology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/cardiology/КР283_2.pdf" ]; then
  echo "Downloading КР283_2.pdf — Гипертрофическая кардиомиопатия"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/cardiology/КР283_2.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=283_2" || echo "FAILED: КР283_2.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=283_2)"
else
  echo "Exists, skipped: КР283_2.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/cardiology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/cardiology/КР746_1.pdf" ]; then
  echo "Downloading КР746_1.pdf — Перикардиты"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/cardiology/КР746_1.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=746_1" || echo "FAILED: КР746_1.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=746_1)"
else
  echo "Exists, skipped: КР746_1.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/cardiology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/cardiology/КР752_1.pdf" ]; then
  echo "Downloading КР752_1.pdf — Нарушения липидного обмена"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/cardiology/КР752_1.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=752_1" || echo "FAILED: КР752_1.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=752_1)"
else
  echo "Exists, skipped: КР752_1.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/cardiology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/cardiology/КР881_1.pdf" ]; then
  echo "Downloading КР881_1.pdf — Митральная недостаточность"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/cardiology/КР881_1.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=881_1" || echo "FAILED: КР881_1.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=881_1)"
else
  echo "Exists, skipped: КР881_1.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/cardiology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/cardiology/КР668_2.pdf" ]; then
  echo "Downloading КР668_2.pdf — Флебит и тромбофлебит поверхностных сосудов"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/cardiology/КР668_2.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=668_2" || echo "FAILED: КР668_2.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=668_2)"
else
  echo "Exists, skipped: КР668_2.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/neurology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/neurology/КР162_3.pdf" ]; then
  echo "Downloading КР162_3.pdf — Головная боль напряжения (ГБН)"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/neurology/КР162_3.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=162_3" || echo "FAILED: КР162_3.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=162_3)"
else
  echo "Exists, skipped: КР162_3.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/neurology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/neurology/КР166_2.pdf" ]; then
  echo "Downloading КР166_2.pdf — Мононевропатии"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/neurology/КР166_2.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=166_2" || echo "FAILED: КР166_2.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=166_2)"
else
  echo "Exists, skipped: КР166_2.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/neurology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/neurology/КР617_5.pdf" ]; then
  echo "Downloading КР617_5.pdf — Когнитивные расстройства у лиц пожилого и старческого возраста"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/neurology/КР617_5.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=617_5" || echo "FAILED: КР617_5.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=617_5)"
else
  echo "Exists, skipped: КР617_5.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/neurology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/neurology/КР778_1.pdf" ]; then
  echo "Downloading КР778_1.pdf — Скелетно-мышечные (неспецифические) боли в нижней части спины"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/neurology/КР778_1.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=778_1" || echo "FAILED: КР778_1.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=778_1)"
else
  echo "Exists, skipped: КР778_1.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/neurology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/neurology/КР780_1.pdf" ]; then
  echo "Downloading КР780_1.pdf — 5q-ассоциированная спинальная мышечная атрофия"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/neurology/КР780_1.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=780_1" || echo "FAILED: КР780_1.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=780_1)"
else
  echo "Exists, skipped: КР780_1.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/neurology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/neurology/КР895_1.pdf" ]; then
  echo "Downloading КР895_1.pdf — Краниальные мононейропатии у взрослых"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/neurology/КР895_1.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=895_1" || echo "FAILED: КР895_1.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=895_1)"
else
  echo "Exists, skipped: КР895_1.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/neurology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/neurology/КР732_1.pdf" ]; then
  echo "Downloading КР732_1.pdf — Очаговая травма головного мозга"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/neurology/КР732_1.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=732_1" || echo "FAILED: КР732_1.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=732_1)"
else
  echo "Exists, skipped: КР732_1.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/neurology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/neurology/КР773_1.pdf" ]; then
  echo "Downloading КР773_1.pdf — Прогрессирующая мышечная дистрофия Дюшенна / Беккера"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/neurology/КР773_1.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=773_1" || echo "FAILED: КР773_1.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=773_1)"
else
  echo "Exists, skipped: КР773_1.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/endocrinology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/endocrinology/КР258_2.pdf" ]; then
  echo "Downloading КР258_2.pdf — Синдром поликистозных яичников"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/endocrinology/КР258_2.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=258_2" || echo "FAILED: КР258_2.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=258_2)"
else
  echo "Exists, skipped: КР258_2.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/endocrinology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/endocrinology/КР329_2.pdf" ]; then
  echo "Downloading КР329_2.pdf — Дифференцированный рак щитовидной железы"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/endocrinology/КР329_2.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=329_2" || echo "FAILED: КР329_2.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=329_2)"
else
  echo "Exists, skipped: КР329_2.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/endocrinology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/endocrinology/КР610_2.pdf" ]; then
  echo "Downloading КР610_2.pdf — Нейроэндокринные опухоли"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/endocrinology/КР610_2.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=610_2" || echo "FAILED: КР610_2.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=610_2)"
else
  echo "Exists, skipped: КР610_2.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/endocrinology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/endocrinology/КР614_2.pdf" ]; then
  echo "Downloading КР614_2.pdf — Патологические переломы, осложняющие остеопороз"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/endocrinology/КР614_2.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=614_2" || echo "FAILED: КР614_2.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=614_2)"
else
  echo "Exists, skipped: КР614_2.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/endocrinology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/endocrinology/КР805_1.pdf" ]; then
  echo "Downloading КР805_1.pdf — Амиодарон-индуцированная дисфункция щитовидной железы"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/endocrinology/КР805_1.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=805_1" || echo "FAILED: КР805_1.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=805_1)"
else
  echo "Exists, skipped: КР805_1.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/endocrinology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/endocrinology/КР82_2.pdf" ]; then
  echo "Downloading КР82_2.pdf — Врожденная дисфункция коры надпочечников (адреногенитальный синдром)"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/endocrinology/КР82_2.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=82_2" || echo "FAILED: КР82_2.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=82_2)"
else
  echo "Exists, skipped: КР82_2.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/endocrinology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/endocrinology/КР87_4.pdf" ]; then
  echo "Downloading КР87_4.pdf — Остеопороз"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/endocrinology/КР87_4.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=87_4" || echo "FAILED: КР87_4.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=87_4)"
else
  echo "Exists, skipped: КР87_4.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/endocrinology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/endocrinology/КР88_4.pdf" ]; then
  echo "Downloading КР88_4.pdf — Первичный гиперпаратиреоз"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/endocrinology/КР88_4.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=88_4" || echo "FAILED: КР88_4.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=88_4)"
else
  echo "Exists, skipped: КР88_4.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/endocrinology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/endocrinology/КР287_3.pdf" ]; then
  echo "Downloading КР287_3.pdf — Сахарный диабет 1 типа у детей"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/endocrinology/КР287_3.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=287_3" || echo "FAILED: КР287_3.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=287_3)"
else
  echo "Exists, skipped: КР287_3.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/gastro"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/gastro/КР176_2.pdf" ]; then
  echo "Downloading КР176_2.pdf — Болезнь Крона"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/gastro/КР176_2.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=176_2" || echo "FAILED: КР176_2.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=176_2)"
else
  echo "Exists, skipped: КР176_2.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/gastro"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/gastro/КР193_2.pdf" ]; then
  echo "Downloading КР193_2.pdf — Язвенный колит"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/gastro/КР193_2.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=193_2" || echo "FAILED: КР193_2.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=193_2)"
else
  echo "Exists, skipped: КР193_2.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/gastro"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/gastro/КР179_3.pdf" ]; then
  echo "Downloading КР179_3.pdf — Дивертикулярная болезнь"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/gastro/КР179_3.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=179_3" || echo "FAILED: КР179_3.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=179_3)"
else
  echo "Exists, skipped: КР179_3.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/gastro"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/gastro/КР273_5.pdf" ]; then
  echo "Downloading КР273_5.pdf — Хронический панкреатит"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/gastro/КР273_5.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=273_5" || echo "FAILED: КР273_5.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=273_5)"
else
  echo "Exists, skipped: КР273_5.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/gastro"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/gastro/КР277_2.pdf" ]; then
  echo "Downloading КР277_2.pdf — Язвенная болезнь"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/gastro/КР277_2.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=277_2" || echo "FAILED: КР277_2.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=277_2)"
else
  echo "Exists, skipped: КР277_2.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/gastro"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/gastro/КР708_2.pdf" ]; then
  echo "Downloading КР708_2.pdf — Гастрит и дуоденит"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/gastro/КР708_2.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=708_2" || echo "FAILED: КР708_2.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=708_2)"
else
  echo "Exists, skipped: КР708_2.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/gastro"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/gastro/КР711_2.pdf" ]; then
  echo "Downloading КР711_2.pdf — Алкогольная болезнь печени"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/gastro/КР711_2.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=711_2" || echo "FAILED: КР711_2.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=711_2)"
else
  echo "Exists, skipped: КР711_2.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/gastro"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/gastro/КР748_2.pdf" ]; then
  echo "Downloading КР748_2.pdf — Неалкогольная жировая болезнь печени"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/gastro/КР748_2.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=748_2" || echo "FAILED: КР748_2.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=748_2)"
else
  echo "Exists, skipped: КР748_2.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/gastro"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/gastro/КР849_1.pdf" ]; then
  echo "Downloading КР849_1.pdf — Грыжа пищеводного отверстия диафрагмы"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/gastro/КР849_1.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=849_1" || echo "FAILED: КР849_1.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=849_1)"
else
  echo "Exists, skipped: КР849_1.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/gastro"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/gastro/КР875_1.pdf" ]; then
  echo "Downloading КР875_1.pdf — Острые кишечные инфекции (ОКИ) у взрослых"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/gastro/КР875_1.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=875_1" || echo "FAILED: КР875_1.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=875_1)"
else
  echo "Exists, skipped: КР875_1.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/gastro"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/gastro/КР878_1.pdf" ]; then
  echo "Downloading КР878_1.pdf — Энтероколит, вызванный Clostridioides difficile (C. difficile)"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/gastro/КР878_1.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=878_1" || echo "FAILED: КР878_1.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=878_1)"
else
  echo "Exists, skipped: КР878_1.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/gastro"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/gastro/КР178_2.pdf" ]; then
  echo "Downloading КР178_2.pdf — Геморрой"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/gastro/КР178_2.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=178_2" || echo "FAILED: КР178_2.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=178_2)"
else
  echo "Exists, skipped: КР178_2.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/therapy"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/therapy/КР469_3.pdf" ]; then
  echo "Downloading КР469_3.pdf — Хроническая болезнь почек (ХБП)"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/therapy/КР469_3.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=469_3" || echo "FAILED: КР469_3.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=469_3)"
else
  echo "Exists, skipped: КР469_3.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/therapy"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/therapy/КР613_2.pdf" ]; then
  echo "Downloading КР613_2.pdf — Старческая астения"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/therapy/КР613_2.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=613_2" || echo "FAILED: КР613_2.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=613_2)"
else
  echo "Exists, skipped: КР613_2.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/therapy"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/therapy/КР615_2.pdf" ]; then
  echo "Downloading КР615_2.pdf — Недостаточность питания (мальнутриция) у пациентов пожилого и старческого возраста"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/therapy/КР615_2.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=615_2" || echo "FAILED: КР615_2.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=615_2)"
else
  echo "Exists, skipped: КР615_2.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/therapy"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/therapy/КР783_1.pdf" ]; then
  echo "Downloading КР783_1.pdf — Гиперчувствительный пневмонит"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/therapy/КР783_1.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=783_1" || echo "FAILED: КР783_1.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=783_1)"
else
  echo "Exists, skipped: КР783_1.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/therapy"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/therapy/КР16_3.pdf" ]; then
  echo "Downloading КР16_3.pdf — Туберкулез у взрослых"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/therapy/КР16_3.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=16_3" || echo "FAILED: КР16_3.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=16_3)"
else
  echo "Exists, skipped: КР16_3.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/therapy"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/therapy/КР79_2.pdf" ]; then
  echo "Downloading КР79_2.pdf — ВИЧ-инфекция у взрослых"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/therapy/КР79_2.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=79_2" || echo "FAILED: КР79_2.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=79_2)"
else
  echo "Exists, skipped: КР79_2.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/therapy"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/therapy/КР838_1.pdf" ]; then
  echo "Downloading КР838_1.pdf — Вирусные пневмонии"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/therapy/КР838_1.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=838_1" || echo "FAILED: КР838_1.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=838_1)"
else
  echo "Exists, skipped: КР838_1.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/therapy"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/therapy/КР891_1.pdf" ]; then
  echo "Downloading КР891_1.pdf — Острый бронхит"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/therapy/КР891_1.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=891_1" || echo "FAILED: КР891_1.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=891_1)"
else
  echo "Exists, skipped: КР891_1.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/therapy"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/therapy/КР898_1.pdf" ]; then
  echo "Downloading КР898_1.pdf — Сепсис (у взрослых)"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/therapy/КР898_1.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=898_1" || echo "FAILED: КР898_1.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=898_1)"
else
  echo "Exists, skipped: КР898_1.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/therapy"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/therapy/КР700_3.pdf" ]; then
  echo "Downloading КР700_3.pdf — Сальмонеллез у взрослых"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/therapy/КР700_3.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=700_3" || echo "FAILED: КР700_3.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=700_3)"
else
  echo "Exists, skipped: КР700_3.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/dermatology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/dermatology/КР234_2.pdf" ]; then
  echo "Downloading КР234_2.pdf — Псориаз"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/dermatology/КР234_2.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=234_2" || echo "FAILED: КР234_2.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=234_2)"
else
  echo "Exists, skipped: КР234_2.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/dermatology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/dermatology/КР238_2.pdf" ]; then
  echo "Downloading КР238_2.pdf — Саркома Капоши"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/dermatology/КР238_2.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=238_2" || echo "FAILED: КР238_2.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=238_2)"
else
  echo "Exists, skipped: КР238_2.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/dermatology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/dermatology/КР751_1.pdf" ]; then
  echo "Downloading КР751_1.pdf — Другие атрофические изменения кожи"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/dermatology/КР751_1.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=751_1" || echo "FAILED: КР751_1.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=751_1)"
else
  echo "Exists, skipped: КР751_1.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/dermatology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/dermatology/КР792_1.pdf" ]; then
  echo "Downloading КР792_1.pdf — Локализованный гипертрихоз"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/dermatology/КР792_1.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=792_1" || echo "FAILED: КР792_1.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=792_1)"
else
  echo "Exists, skipped: КР792_1.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/dermatology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/dermatology/КР796_1.pdf" ]; then
  echo "Downloading КР796_1.pdf — Негонококковый (неспецифический) уретрит у мужчин"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/dermatology/КР796_1.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=796_1" || echo "FAILED: КР796_1.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=796_1)"
else
  echo "Exists, skipped: КР796_1.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/dermatology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/dermatology/КР220_2.pdf" ]; then
  echo "Downloading КР220_2.pdf — Контагиозный моллюск"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/dermatology/КР220_2.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=220_2" || echo "FAILED: КР220_2.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=220_2)"
else
  echo "Exists, skipped: КР220_2.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/dermatology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/dermatology/КР200_3.pdf" ]; then
  echo "Downloading КР200_3.pdf — Эритразма"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/dermatology/КР200_3.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=200_3" || echo "FAILED: КР200_3.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=200_3)"
else
  echo "Exists, skipped: КР200_3.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/dermatology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/dermatology/КР241_3.pdf" ]; then
  echo "Downloading КР241_3.pdf — Урогенитальный трихомониаз"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/dermatology/КР241_3.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=241_3" || echo "FAILED: КР241_3.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=241_3)"
else
  echo "Exists, skipped: КР241_3.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/dermatology"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/dermatology/КР194_2.pdf" ]; then
  echo "Downloading КР194_2.pdf — Хламидийная инфекция"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/dermatology/КР194_2.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=194_2" || echo "FAILED: КР194_2.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=194_2)"
else
  echo "Exists, skipped: КР194_2.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/ other"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/ other/КР396_3.pdf" ]; then
  echo "Downloading КР396_3.pdf — Злокачественное новообразование ободочной кишки"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/ other/КР396_3.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=396_3" || echo "FAILED: КР396_3.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=396_3)"
else
  echo "Exists, skipped: КР396_3.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/ other"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/ other/КР574_1.pdf" ]; then
  echo "Downloading КР574_1.pdf — Рак желудка"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/ other/КР574_1.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=574_1" || echo "FAILED: КР574_1.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=574_1)"
else
  echo "Exists, skipped: КР574_1.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/ other"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/ other/КР584_2.pdf" ]; then
  echo "Downloading КР584_2.pdf — Герминогенные опухоли у мужчин"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/ other/КР584_2.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=584_2" || echo "FAILED: КР584_2.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=584_2)"
else
  echo "Exists, skipped: КР584_2.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/ other"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/ other/КР811_1.pdf" ]; then
  echo "Downloading КР811_1.pdf — Прочие первичные грыжи брюшной стенки"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/ other/КР811_1.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=811_1" || echo "FAILED: КР811_1.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=811_1)"
else
  echo "Exists, skipped: КР811_1.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/ other"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/ other/КР844_1.pdf" ]; then
  echo "Downloading КР844_1.pdf — Инфекция, ассоциированная с ортопедическими имплантатами"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/ other/КР844_1.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=844_1" || echo "FAILED: КР844_1.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=844_1)"
else
  echo "Exists, skipped: КР844_1.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/ other"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/ other/КР7_2.pdf" ]; then
  echo "Downloading КР7_2.pdf — Мочекаменная болезнь"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/ other/КР7_2.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=7_2" || echo "FAILED: КР7_2.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=7_2)"
else
  echo "Exists, skipped: КР7_2.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/ other"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/ other/КР6_2.pdf" ]; then
  echo "Downloading КР6_2.pdf — Доброкачественная гиперплазия предстательной железы"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/ other/КР6_2.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=6_2" || echo "FAILED: КР6_2.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=6_2)"
else
  echo "Exists, skipped: КР6_2.pdf"
fi

mkdir -p "$BASE_DIR/docs/01_raw/minzdrav/ other"
if [ ! -s "$BASE_DIR/docs/01_raw/minzdrav/ other/КР400_2.pdf" ]; then
  echo "Downloading КР400_2.pdf — Хронический болевой синдром у взрослых пациентов, нуждающихся в паллиативной медицинской помощи"
  curl -L --fail --retry 3 --connect-timeout 20 -o "$BASE_DIR/docs/01_raw/minzdrav/ other/КР400_2.pdf" "https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=400_2" || echo "FAILED: КР400_2.pdf (https://apicr.minzdrav.gov.ru/api.ashx?op=GetClinrecPdf&id=400_2)"
else
  echo "Exists, skipped: КР400_2.pdf"
fi

