# СТРАТЕГИЯ СВЕДЕНИЯ И МАСТЕРИНГА

## АНАЛИЗ ПРОЕКТА

### Структура трека
| Файл | Роль | Каналы | Sample Rate | Bit Depth | Длительность | Peak dBFS | RMS dBFS |
|------|------|--------|-------------|-----------|--------------|-----------|----------|
| 1 дорожка.wav | Лид Вокал | Mono | 44.1 kHz | 24-bit | 13.84 сек | -8.4 | -25.6 |
| 2 дорожка_дабл_лида.wav | Дабл Лида | Mono | 44.1 kHz | 24-bit | 14.40 сек | -4.3 | -31.6 |
| 3 дороожка_звуки_у-у-у.wav | Бэки "у-у-у" | Mono | 44.1 kHz | 24-bit | 14.33 сек | -7.2 | -29.8 |
| 4 дорожка_шу_шу_еа.wav | Бэки "шу шу еа" | Mono | 44.1 kHz | 24-bit | 14.47 сек | -11.9 | -36.5 |
| 5 дорожка_все_что_было_важно.wav | Вокал партия | Mono | 44.1 kHz | 24-bit | 14.30 сек | -4.3 | -32.9 |
| Untitled (3).wav | Инструментал | Stereo | 48 kHz | 16-bit | 2:03 | -3.4 | -17.0 |

### Выявленные проблемы

1. **Несоответствие Sample Rate**: Вокалы 44.1 kHz, инструментал 48 kHz → конвертировать всё в 48 kHz
2. **Несоответствие Bit Depth**: Вокалы 24-bit, инструментал 16-bit → работать в 24-bit/48 kHz
3. **Лид вокал (дорожка 1)**: 82% энергии в басах — сильный proximity effect, mud
4. **Бэки "у-у-у"**: 93.6% в мидах — слишком узкий спектр
5. **Инструментал**: 76.7% энергии в sub-bass + bass — нужно вырезать место для вокала

---

## ЭТАП 1: ПОДГОТОВКА СЕССИИ

### 1.1 Настройка проекта DAW
```
Project Settings:
- Sample Rate: 48 kHz
- Bit Depth: 24-bit (32-bit float для обработки)
- Buffer Size: 256-512 samples
- Dither: отключить до мастеринга
```

### 1.2 Конвертация файлов
- Вокальные дорожки: upsample 44.1 → 48 kHz (высококачественный алгоритм)
- Инструментал: upsample 16-bit → 24-bit (для headroom при обработке)

### 1.3 Организация дорожек
```
Routing Structure:
├── BUS: VOCALS
│   ├── LEAD VOCAL (дорожка 1)
│   ├── DOUBLE (дорожка 2)
│   ├── BUS: BACKS
│   │   ├── У-у-у (дорожка 3)
│   │   ├── Шу-шу-еа (дорожка 4)
│   │   └── Все что было важно (дорожка 5)
├── BUS: INSTRUMENTAL
│   └── Untitled (3).wav
├── BUS: EFFECTS (REVERB, DELAY)
└── MASTER BUS
```

---

## ЭТАП 2: ОБРАБОТКА ДОРОЖЕК

### 2.1 ЛИД ВОКАЛ (дорожка 1) — ГЛАВНЫЙ ПРИОРИТЕТ

**ПРОБЛЕМА**: 82% энергии в басах, недостаточно air и presence

**Цепочка эффектов (Insert Chain):**

```
1. NOISE GATE (если нужно)
   - Threshold: -45 dB
   - Attack: 0.5 ms
   - Release: 50 ms
   - Range: -inf (или soft gate -20 dB)

2. HIGH-PASS FILTER (КРИТИЧНО!)
   - Частота: 80-100 Hz
   - Slope: 18 dB/oct
   - Убирает mud и proximity effect

3. DE-ESSER
   - Frequency: 5-8 kHz
   - Threshold: -30 dB
   - Ratio: 4:1
   - Только если есть сибилянты

4. КОМПРЕССОР #1 (Taming Peaks)
   - Ratio: 4:1
   - Attack: 10-15 ms (пропускает транзиенты)
   - Release: Auto или 80-120 ms
   - Threshold: -18 dB
   - Gain reduction: 3-6 dB
   - Рекомендуемые плагины: LA-2A, CLA-2A, TDR Kotelnikov

5. EQ (Формирование звука)
   - HPF: 80 Hz (уже сделано)
   - CUT: -3 dB @ 200-300 Hz (Q=2) — убрать boominess
   - CUT: -2 dB @ 400-500 Hz (Q=3) — убрать boxiness
   - BOOST: +2 dB @ 2-4 kHz (Q=1.5) — presence, разборчивость
   - BOOST: +3 dB @ 8-12 kHz (Q=0.7 shelf) — air, блеск

6. КОМПРЕССОР #2 (Glue/Character)
   - Ratio: 2:1
   - Attack: 30 ms
   - Release: 100 ms
   - Gain reduction: 2-3 dB
   - Рекомендуемые: 1176, Distressor

7. САТУРАЦИЯ (опционально, для тепла)
   - Tape saturation: очень легкая (Softube Tape, Waves J37)
   - Drive: 1-2 dB
```

**Панорамирование**: CENTER (0)

**Fader Level**: Начать с -6 dB, настроить по миксу

---

### 2.2 ДАБЛ ЛИДА (дорожка 2)

**РОЛЬ**: Поддержка лид вокала, создание ширины и плотности

**Цепочка эффектов:**

```
1. HIGH-PASS FILTER
   - Частота: 120 Hz (выше чем лид!)
   - Slope: 12 dB/oct

2. EQ
   - CUT: -4 dB @ 200-400 Hz — освободить место для лида
   - CUT: -2 dB @ 2-3 kHz — не конкурировать с presence лида
   - BOOST: +2 dB @ 5-6 kHz — добавить отличие

3. КОМПРЕССОР
   - Ratio: 3:1
   - Attack: 20 ms
   - Release: Auto
   - Gain reduction: 4-6 dB (сильнее чем лид — более ровный)

4. STEREO WIDENER / MICRO-PITCH
   - Waves Doubler или аналог
   - Detune: +7 cents L / -7 cents R
   - Delay: 10-15 ms L / 20-25 ms R
   - ИЛИ просто панорамировать L/R при дублировании
```

**Панорамирование**:
- Вариант 1: Дабл один → L30, дублировать → R30
- Вариант 2: Micro-pitch shift для стерео ширины

**Fader Level**: -8 dB ниже лида (поддержка, не конкуренция)

---

### 2.3 БЭКИ "У-У-У" (дорожка 3)

**ПРОБЛЕМА**: 93.6% энергии в мидах — слишком узко

**Цепочка эффектов:**

```
1. HIGH-PASS FILTER
   - Частота: 150 Hz
   - Slope: 12 dB/oct

2. EQ
   - CUT: -3 dB @ 300-500 Hz — уменьшить mud
   - CUT: -2 dB @ 800 Hz-1.2 kHz — освободить место для лида
   - BOOST: +4 dB @ 3-5 kHz — добавить presence
   - BOOST: +3 dB @ 10 kHz shelf — добавить air

3. КОМПРЕССОР
   - Ratio: 4:1
   - Attack: 5 ms (быстрый — бэки должны быть ровными)
   - Release: 60 ms
   - Gain reduction: 6-8 dB

4. REVERB SEND (см. раздел эффектов)
   - Plate reverb: -12 dB send
   - Создаёт глубину и пространство

5. WIDENER
   - Haas effect: 15-25 ms delay на одном канале
   - Или stereo spread plugin
```

**Панорамирование**: L40 / R40 (широко, за лид вокалом)

**Fader Level**: -10 dB относительно лида

---

### 2.4 БЭКИ "ШУ-ШУ ЕА" (дорожка 4)

**ХАРАКТЕРИСТИКА**: Больше high-mids (16.9%) — консонантные звуки

**Цепочка эффектов:**

```
1. HIGH-PASS FILTER
   - Частота: 200 Hz
   - Slope: 18 dB/oct

2. DE-ESSER (ВАЖНО для "ш" звуков!)
   - Frequency: 6-10 kHz
   - Threshold: -25 dB
   - Ratio: 6:1

3. EQ
   - CUT: -4 dB @ 2.5-3.5 kHz (Q=2) — убрать резкость
   - BOOST: +2 dB @ 6-8 kHz — сохранить чёткость
   - BOOST: +2 dB @ 12 kHz shelf — air

4. КОМПРЕССОР
   - Ratio: 6:1 (сильная компрессия)
   - Attack: 2 ms (очень быстрый)
   - Release: 50 ms
   - Gain reduction: 8-10 dB

5. REVERB SEND
   - Короткий room: -8 dB send
   - Pre-delay: 30 ms
```

**Панорамирование**: L60 / R60 (самые широкие)

**Fader Level**: -12 dB относительно лида

---

### 2.5 ВОКАЛ "ВСЁ ЧТО БЫЛО ВАЖНО" (дорожка 5)

**ХАРАКТЕРИСТИКА**: 84.9% мидов — мелодическая партия

**Цепочка эффектов:**

```
1. HIGH-PASS FILTER
   - Частота: 100 Hz
   - Slope: 12 dB/oct

2. EQ
   - CUT: -3 dB @ 250-400 Hz
   - CUT: -2 dB @ 1-2 kHz — не конкурировать с лидом
   - BOOST: +3 dB @ 4-5 kHz
   - BOOST: +2 dB @ 10 kHz shelf

3. КОМПРЕССОР
   - Ratio: 3:1
   - Attack: 15 ms
   - Release: Auto
   - Gain reduction: 4-6 dB

4. REVERB/DELAY SEND
   - Medium plate: -10 dB
```

**Панорамирование**: L20 / R20 или CENTER (в зависимости от аранжировки)

**Fader Level**: -8 dB относительно лида

---

## ЭТАП 3: ОБРАБОТКА ИНСТРУМЕНТАЛА

### 3.1 ПРОБЛЕМА
- 76.7% энергии в низких частотах (sub-bass + bass)
- Мало места для вокала в mid-range

### 3.2 Цепочка эффектов

```
1. EQ (Carving Space for Vocals)
   - CUT: -2 dB @ 200-400 Hz (wide Q=0.8) — освободить low-mids
   - CUT: -3 dB @ 2-4 kHz (Q=1.5) — освободить presence zone для вокала
   - BOOST: +1 dB @ 60-80 Hz — подчеркнуть sub-bass
   - BOOST: +1.5 dB @ 8-12 kHz — добавить shimmer

2. SIDECHAIN COMPRESSION (КРЕАТИВНАЯ ФИШКА!)
   - Компрессор на инструментале
   - Sidechain input: Лид вокал
   - Ratio: 2:1 - 3:1
   - Attack: 10 ms
   - Release: 100-150 ms
   - Threshold: настроить на -3 dB gain reduction
   - ЭФФЕКТ: Инструментал "дышит" под вокал

3. MID-SIDE EQ (продвинутая техника)
   - MID: Cut -2 dB @ 1-3 kHz (освободить центр для вокала)
   - SIDE: Boost +2 dB @ 8-12 kHz (расширить стерео)

4. MULTIBAND COMPRESSION (опционально)
   - Low band (20-200 Hz): Ratio 4:1
   - Mid band (200-2000 Hz): Ratio 2:1
   - High band (2000-20000 Hz): Ratio 2:1
```

---

## ЭТАП 4: ШИНЫ И ГРУППОВАЯ ОБРАБОТКА

### 4.1 VOCAL BUS

```
1. BUS COMPRESSION (Glue)
   - SSL Bus Comp или аналог
   - Ratio: 2:1
   - Attack: 30 ms
   - Release: Auto
   - Threshold: -2 до -4 dB gain reduction

2. PARALLEL COMPRESSION
   - Отправить вокал бас на параллельный канал
   - Компрессор: 10:1, Attack 1ms, Release 50ms
   - Сильно сжать (10-15 dB GR)
   - Подмешать -12 dB к основному
   - ЭФФЕКТ: Агрессия + punch без потери динамики

3. BUS EQ
   - Subtle boost +1 dB @ 10 kHz shelf
   - Subtle cut -1 dB @ 300 Hz

4. TAPE SATURATION
   - Очень лёгкая (Studer, Ampex эмуляция)
   - 1-2 dB saturation
```

### 4.2 BACKS BUS

```
1. BUS COMPRESSION
   - Ratio: 4:1
   - Fast attack/release
   - Heavy gain reduction: 6-8 dB
   - ЦЕЛЬ: Максимально ровные бэки

2. STEREO WIDENER
   - Небольшое расширение стерео базы
   - Или mid-side: reduce mid, boost sides

3. LOW-PASS FILTER
   - 14-16 kHz
   - Убрать "воздух" из бэков (оставить для лида)
```

---

## ЭТАП 5: ЭФФЕКТЫ (SENDS)

### 5.1 REVERB #1: PLATE (Основной вокальный)

```
Plugin: Valhalla Plate, Soundtoys Little Plate, UAD EMT 140
Settings:
- Decay: 1.8-2.2 sec
- Pre-delay: 40-60 ms (сохраняет чёткость)
- Damping: High frequencies rolled off @ 6 kHz
- Mix: 100% wet (это send!)
- EQ на reverb return: HPF 300 Hz, LPF 8 kHz
```

**Send Levels:**
- Лид вокал: -14 dB
- Дабл: -12 dB
- Бэки: -10 dB (больше reverb = дальше в миксе)

### 5.2 REVERB #2: ROOM (Для естественности)

```
Plugin: Valhalla Room, Lexicon, Altiverb
Settings:
- Room type: Medium room / Studio
- Decay: 0.6-0.8 sec
- Pre-delay: 10-20 ms
- Early reflections: 40-50%
```

**Send Levels:**
- Лид вокал: -18 dB (чуть-чуть)
- Инструментал: -20 dB (cohesion)

### 5.3 DELAY #1: SLAP (Rhythm)

```
Plugin: EchoBoy, H-Delay, ReaDelay
Settings:
- Time: 80-120 ms (или sync to tempo: 1/16)
- Feedback: 0-10%
- Mix: 100% wet
- EQ: HPF 400 Hz, LPF 4 kHz (тёмный delay)
```

**Send Levels:**
- Лид вокал: -16 dB
- Применять на акцентах и окончаниях фраз

### 5.4 DELAY #2: PING-PONG (Креатив)

```
Settings:
- Time: 1/8 или 1/4 (sync to BPM)
- Feedback: 30-40%
- Pan: L/R ping-pong
- Filter: Heavy HPF, LPF
```

**Send Levels:**
- Бэки: -12 dB
- Создаёт движение и интерес

### 5.5 CREATIVE: THROW (Эффект для драматических моментов)

```
Plugin: Любой delay/reverb с автоматизацией
- Автоматизировать send на последние слова фраз
- Send level: 0 dB на момент (громкий throw)
- Затем fade out
```

---

## ЭТАП 6: БАЛАНС И АВТОМАТИЗАЦИЯ

### 6.1 Статический баланс (отправная точка)

```
Fader Positions (относительно):
├── INSTRUMENTAL:     0 dB (reference)
├── LEAD VOCAL:      +2 to +4 dB (должен быть впереди)
├── DOUBLE:          -4 dB от лида
├── BACKS:           -8 to -10 dB от лида
└── EFFECTS RETURNS: -12 to -18 dB
```

### 6.2 Автоматизация (КЛЮЧ К ПРОФЕССИОНАЛЬНОМУ ЗВУКУ!)

```
ГРОМКОСТЬ ЛИДА:
- Поднять тихие слоги/слова (+2-3 dB)
- Опустить громкие (+1-2 dB down)
- Цель: Разница не более 3-4 dB в perceived loudness

ЭФФЕКТЫ:
- Больше reverb в тихих частях
- Меньше reverb в громких/busy частях
- Delay throws на окончаниях фраз

ПАНОРАМА:
- Автоматизировать бэки для движения
- Можно делать subtle panning automation

EQ:
- Автоматизировать presence boost в ключевых моментах
```

---

## ЭТАП 7: МАСТЕРИНГ

### 7.1 Pre-Master Checklist
- [ ] Все клипы убраны
- [ ] Peak headroom: -3 to -6 dBFS
- [ ] Нет DC offset
- [ ] Все автоматизации завершены
- [ ] Экспорт в 24-bit/48 kHz (или выше)

### 7.2 Мастеринг цепочка

```
1. CORRECTIVE EQ (если нужно)
   - Тонкие коррекции на основе референса
   - Linear phase EQ для прозрачности

2. MULTIBAND COMPRESSION
   - Low (20-200 Hz): 2:1, slow attack
   - Mid (200-2000 Hz): 2:1, medium attack
   - High (2000+ Hz): 1.5:1, fast attack
   - Total GR: 2-4 dB max

3. STEREO ENHANCEMENT
   - Subtle mid-side processing
   - Widen highs (above 5 kHz)
   - Keep bass in mono (below 150 Hz)

4. HARMONIC EXCITEMENT
   - Tape saturation или tube warmth
   - Очень subtle: 0.5-1 dB

5. LIMITER
   - Ceiling: -1.0 dBTP (True Peak)
   - Target LUFS: -14 для стриминга (Spotify/Apple Music)
   - или -9 to -11 LUFS для более громкого мастера
   - Attack: 0.1-1 ms
   - Release: Auto или 50-100 ms
   - Max gain reduction: 3-4 dB
```

### 7.3 Рекомендуемые плагины для мастеринга
- **EQ**: FabFilter Pro-Q 3, Ozone EQ
- **Compression**: Ozone Dynamics, Shadow Hills
- **Limiter**: FabFilter Pro-L 2, Ozone Maximizer
- **Metering**: Youlean Loudness Meter (FREE), iZotope Insight

---

## ЭТАП 8: КРЕАТИВНЫЕ ФИШКИ

### 8.1 ТОПОВЫЕ ПРИЁМЫ

1. **Vocal Doubling Effect (без дабла)**
   ```
   Micro-pitch shift: +7 cents L, -7 cents R
   Delay: 10ms L, 20ms R
   Mix: 30% wet
   ```

2. **Telephone Effect (для акцента)**
   ```
   Bandpass filter: 500 Hz - 3 kHz
   Saturation: heavy
   Применять на определённых словах через автоматизацию
   ```

3. **Reverse Reverb (на ключевых словах)**
   ```
   - Экспорт слова
   - Reverse
   - Добавить heavy reverb
   - Reverse снова
   - Поставить перед словом
   ```

4. **Vocal Chops (современный звук)**
   ```
   - Нарезать бэки на слоги
   - Квантизировать по сетке
   - Pitch shift разные части
   - Добавить stutter/glitch эффекты
   ```

5. **Formant Shifting**
   ```
   Plugin: Soundtoys Little AlterBoy, Waves SoundShifter
   - Shift формант для интересного тембра
   - Можно автоматизировать
   ```

6. **Sidechain Pumping (современный поп/EDM)**
   ```
   - Kick → sidechain на весь микс (кроме kick)
   - Subtle pumping 1-2 dB
   - Создаёт groove и энергию
   ```

7. **Layered Harmonies**
   ```
   - Pitch-shift бэков на терции/квинты
   - Mix очень тихо (-18 dB)
   - Создаёт богатство без явных гармоний
   ```

8. **Lo-Fi Layer**
   ```
   - Параллельная обработка
   - Bitcrusher + heavy saturation
   - LPF @ 3 kHz
   - Mix: 5-10%
   - Добавляет "грязь" и характер
   ```

---

## РЕКОМЕНДУЕМЫЕ ПЛАГИНЫ

### БЕСПЛАТНЫЕ (топ качество)
| Категория | Плагин | Описание |
|-----------|--------|----------|
| EQ | TDR Nova | Динамический EQ |
| Compressor | TDR Kotelnikov | Прозрачный компрессор |
| Saturation | Softube Saturation Knob | Простая сатурация |
| Reverb | Valhalla Supermassive | Креативный reverb |
| Limiter | Youlean Loudness Meter | Измерение LUFS |
| De-Esser | Analog Obsession LALA | LA-2A эмуляция |

### ПЛАТНЫЕ (индустриальный стандарт)
| Категория | Плагин |
|-----------|--------|
| EQ | FabFilter Pro-Q 3 |
| Compressor | Waves CLA-2A, CLA-76 |
| Vocal Chain | Waves Vocal Rider |
| Reverb | Valhalla Vintage Verb, FabFilter Pro-R |
| De-Esser | FabFilter Pro-DS |
| Saturation | Soundtoys Decapitator |
| Mastering | iZotope Ozone 11 |

---

## ФИНАЛЬНЫЙ ЧЕКЛИСТ

### Перед экспортом
- [ ] A/B сравнение с референсом
- [ ] Проверка на разных системах (наушники, колонки, телефон, машина)
- [ ] Mono compatibility check
- [ ] Проверка начала и конца трека (fade in/out)
- [ ] Удаление всех соло/мьют
- [ ] Проверка метаданных

### Форматы экспорта
```
Master Files:
- WAV 24-bit/48 kHz (архив)
- WAV 16-bit/44.1 kHz (CD quality)

Streaming:
- MP3 320 kbps
- AAC 256 kbps
- или WAV/FLAC (Spotify принимает lossless)
```

---

## WORKFLOW: ШАГ ЗА ШАГОМ

```
1. [ ] Настроить проект (48 kHz, 24-bit)
2. [ ] Конвертировать все файлы
3. [ ] Организовать routing (buses)
4. [ ] Gain staging (все дорожки ~-18 dBFS RMS)
5. [ ] Обработать лид вокал (ПЕРВЫЙ ПРИОРИТЕТ)
6. [ ] Обработать инструментал (sidechain, EQ carving)
7. [ ] Обработать дабл
8. [ ] Обработать бэки
9. [ ] Настроить sends (reverb, delay)
10. [ ] Статический баланс
11. [ ] Автоматизация громкости
12. [ ] Автоматизация эффектов
13. [ ] A/B с референсом
14. [ ] Финальные твики
15. [ ] Экспорт pre-master
16. [ ] Мастеринг
17. [ ] Финальная проверка
18. [ ] Экспорт всех форматов
```

---

*Документ создан на основе анализа проекта. Удачи со сведением!*
