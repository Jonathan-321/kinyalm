# KinyaLM Recovery Task Bank

This bank is reserved for native-speaker evaluation. None of these prompts may
be copied, paraphrased, or used as source material for SFT or continued
pretraining. Reviewers should flag any prompt that is itself unnatural or
ambiguous before scoring a model response.

## Held-Out Tasks

| ID | Category | Split | Learner Prompt | Review Focus |
| --- | --- | --- | --- | --- |
| T1001 | Greeting and introduction | benchmark-only | Sobanura igihe umuntu akoresha `Muraho` n'igihe akoresha `Mwaramutse`. | correctness, time of day, natural register |
| T1002 | Greeting and introduction | benchmark-only | How would you politely greet an older person you are meeting for the first time in the afternoon? Answer in Kinyarwanda and explain briefly. | politeness, natural wording, explanation |
| T1003 | Greeting and introduction | benchmark-only | Komeza iki kiganiro mu buryo busanzwe: A: Muraho. B: Muraho neza. | conversational naturalness, turn taking |
| T1004 | Greeting and introduction | benchmark-only | Hindura mu Kinyarwanda: Hello, my name is Keza. It is nice to meet you. | name construction, natural introduction |
| T1005 | Greeting and introduction | benchmark-only | A learner says `Amakuru meza` when somebody asks `Amakuru?`. Is that natural? Correct or explain it. | idiomatic response, correction quality |
| T1006 | Greeting and introduction | benchmark-only | Tangira ikiganiro kigufi hagati y'umunyeshuri mushya n'umwarimu. | beginner language, respectful dialogue |
| T1007 | Greeting and introduction | benchmark-only | Explain the difference between `Wiriwe` and `Mwiriwe` to a beginner. | number or respect distinction, clarity |
| T1008 | Greeting and introduction | benchmark-only | Nabaza nte umuntu izina rye mu buryo bwiyubashye? | respectful question formation |
| T1009 | Greeting and introduction | benchmark-only | Give two natural ways to answer someone who says `Murakoze cyane`. | idiomatic replies, register |
| T1010 | Greeting and introduction | benchmark-only | Correct this introduction and explain the problem: `Jyewe yitwa Aline.` | subject agreement, name construction |
| T1011 | Vocabulary and usage | benchmark-only | Sobanura ijambo `ubupfura` kandi utange ingero ebyiri zisanzwe. | definition, natural examples |
| T1012 | Vocabulary and usage | benchmark-only | Ni irihe tandukaniro riri hagati ya `kubona` na `kureba`? | semantic distinction, examples |
| T1013 | Vocabulary and usage | benchmark-only | Explain `umuryango` in English and show two different meanings it can have in Kinyarwanda. | polysemy, context accuracy |
| T1014 | Vocabulary and usage | benchmark-only | Sobanura amagambo `akazi`, `umwuga`, na `umurimo` uko atandukanye. | semantic nuance, natural usage |
| T1015 | Vocabulary and usage | benchmark-only | Teach a beginner the words for teacher, student, classroom, book, and lesson. Include one short example. | vocabulary correctness, beginner fit |
| T1016 | Vocabulary and usage | benchmark-only | Ijambo `gusoma` rishobora kugira ibisobanuro birenze kimwe? Tanga interuro zisobanutse. | polysemy, sentence correctness |
| T1017 | Vocabulary and usage | benchmark-only | Explain when `iwacu` means my home, our home, or my home area. | pronoun and context nuance |
| T1018 | Vocabulary and usage | benchmark-only | Mpa amagambo ane ajyanye n'imvura, hanyuma uyakoreshe mu gika kigufi. | vocabulary, agreement, paragraph coherence |
| T1019 | Vocabulary and usage | benchmark-only | What is a natural Kinyarwanda word or phrase for a classmate? State uncertainty if usage varies. | lexical accuracy, uncertainty |
| T1020 | Vocabulary and usage | benchmark-only | Sobanura itandukaniro riri hagati ya `inshuti` na `mugenzi`. | semantic nuance, examples |
| T1021 | Translation EN-RW | benchmark-only | Translate naturally into Kinyarwanda: I am a student, but my sister is a teacher. | noun classes, coordination |
| T1022 | Translation EN-RW | benchmark-only | Translate naturally into Kinyarwanda: Tomorrow we will go to school early. | future tense, adverb placement |
| T1023 | Translation EN-RW | benchmark-only | Translate naturally into Kinyarwanda: Please explain that word again more slowly. | politeness, classroom language |
| T1024 | Translation EN-RW | benchmark-only | Translate naturally into Kinyarwanda: The children are reading two new books. | class agreement, number, adjective |
| T1025 | Translation EN-RW | benchmark-only | Translate naturally into Kinyarwanda: I did not see him because it was raining. | negation, object reference, cause |
| T1026 | Translation EN-RW | benchmark-only | Translate naturally into Kinyarwanda: My parents live in Kigali, but I study in Huye. | possessive agreement, location |
| T1027 | Translation EN-RW | benchmark-only | Translate naturally into Kinyarwanda: Could you show me where the library is? | polite request, location phrase |
| T1028 | Translation EN-RW | benchmark-only | Translate naturally into Kinyarwanda: We have already finished today's lesson. | aspect, possessive or temporal form |
| T1029 | Translation EN-RW | benchmark-only | Translate naturally into Kinyarwanda: If I make a mistake, please correct me. | conditional, object agreement, politeness |
| T1030 | Translation EN-RW | benchmark-only | Translate naturally into Kinyarwanda: She gave the student a pen and asked him to write. | subject continuity, objects, reported action |
| T1031 | Translation RW-EN | benchmark-only | Translate into natural English: `Nubwo yari ananiwe, yakomeje kwiga.` | concessive clause, tense |
| T1032 | Translation RW-EN | benchmark-only | Translate into natural English: `Abana bato barimo gukinira hanze.` | progressive meaning, location |
| T1033 | Translation RW-EN | benchmark-only | Translate into natural English: `Nari nibagiwe ko uyu munsi dufite isuzuma.` | tense, embedded clause |
| T1034 | Translation RW-EN | benchmark-only | Translate into natural English: `Mumbabarire, sinumvise neza icyo mwavuze.` | respectful address, natural English |
| T1035 | Translation RW-EN | benchmark-only | Translate into natural English: `Igitabo nagutije wakirangije?` | object relation, question meaning |
| T1036 | Translation RW-EN | benchmark-only | Translate into natural English: `Iyo mbonye umwanya, nkunda gusoma inkuru.` | habitual conditional, natural phrasing |
| T1037 | Translation RW-EN | benchmark-only | Translate into natural English: `Twari kuhagera kare iyo imodoka itadupfiraho.` | counterfactual meaning, idiom |
| T1038 | Translation RW-EN | benchmark-only | Translate into natural English: `Uwo mwana asa na se ariko avuga nka nyina.` | comparison, possessives |
| T1039 | Translation RW-EN | benchmark-only | Translate into natural English: `Sinzi niba aza uyu munsi cyangwa ejo.` | uncertainty, time ambiguity |
| T1040 | Translation RW-EN | benchmark-only | Translate into natural English: `Bamubwiye ko atagomba gutinda.` | reported speech, negation |
| T1041 | Sentence correction | benchmark-only | Kosora kandi usobanure: `Ejo nzagiye ku ishuri.` | future construction, explanation |
| T1042 | Sentence correction | benchmark-only | Kosora kandi usobanure: `Ndi kwiga Ikinyarwanda buri munsi.` | aspect, natural present form |
| T1043 | Sentence correction | benchmark-only | Kosora kandi usobanure: `Abana muto arakina.` | noun-class agreement |
| T1044 | Sentence correction | benchmark-only | Kosora kandi usobanure: `Ibitabo nshya iri ku meza.` | plural agreement, locative phrase |
| T1045 | Sentence correction | benchmark-only | Kosora kandi usobanure: `Jyewe akunda gusoma.` | subject agreement |
| T1046 | Sentence correction | benchmark-only | Kosora kandi usobanure: `Umwarimu bafasha umunyeshuri.` | subject-verb agreement |
| T1047 | Sentence correction | benchmark-only | Correct the Kinyarwanda sentence and explain in English: `Nabonye we ejo.` | object pronoun placement |
| T1048 | Sentence correction | benchmark-only | Correct the Kinyarwanda sentence and explain in English: `Mfite imyaka makumyabiri na umwe.` | numeral agreement, age expression |
| T1049 | Sentence correction | benchmark-only | Kosora interuro niba ikeneye gukosorwa: `Ntabwo sinagiye ku kazi.` | double negation, careful judgment |
| T1050 | Sentence correction | benchmark-only | Kosora kandi usobanure: `Uyu abakobwa ni abanyeshuri.` | demonstrative and noun agreement |
| T1051 | Sentence correction | benchmark-only | A learner wrote `Ndashaka ko kugenda ubu.` Correct it and explain the clause structure. | complement clause, infinitive use |
| T1052 | Sentence correction | benchmark-only | A learner wrote `Yagiye muri Kigali.` Correct it if necessary and explain the location marker. | locative selection, uncertainty |
| T1053 | Sentence correction | benchmark-only | A learner wrote `Nshobora kuvuga Ikinyarwanda make.` Give a natural correction. | quantifier or adverb choice |
| T1054 | Sentence correction | benchmark-only | A learner wrote `Twebwe ni inshuti.` Correct it and explain agreement. | copular agreement, pronoun |
| T1055 | Sentence correction | benchmark-only | A learner wrote `Umwana zanjye bariga.` Correct all agreement errors. | possessive, number, verb agreement |
| T1056 | Sentence correction | benchmark-only | A learner wrote `Nagiye ku rugo.` Decide whether it is natural and explain alternatives. | locative idiom, nuance |
| T1057 | Sentence correction | benchmark-only | A learner wrote `Ndabona televiziyo buri mugoroba.` Correct it if the intended meaning is watching television. | lexical choice, habitual aspect |
| T1058 | Sentence correction | benchmark-only | A learner wrote `Mwaramutse` at 8 p.m. Respond like a tutor and correct the usage. | time-sensitive greeting, tutoring tone |
| T1059 | Sentence correction | benchmark-only | A learner translated `I miss you` word for word. Give a natural Kinyarwanda expression and warn if context matters. | nonliteral translation, uncertainty |
| T1060 | Sentence correction | benchmark-only | A learner says `Uri nde?` to an elder. Explain whether grammar and register are both appropriate. | grammar versus politeness |
| T1061 | Morphology and grammar | benchmark-only | Explain the parts of `Sinzabikora` without inventing a rule. | morpheme segmentation, accuracy |
| T1062 | Morphology and grammar | benchmark-only | Gereranya `umuntu`, `abantu`, `umwana`, na `abana` usobanure impinduka. | noun-class singular and plural |
| T1063 | Morphology and grammar | benchmark-only | Explain why the adjective changes in `umwana muto` and `abana bato`. | adjective agreement |
| T1064 | Morphology and grammar | benchmark-only | Explain the agreement pattern in `Ibitabo bibiri bishya birahari.` | class agreement across words |
| T1065 | Morphology and grammar | benchmark-only | Break down `narabibonye` into meaningful parts for an intermediate learner. | tense, object marker, verb stem |
| T1066 | Morphology and grammar | benchmark-only | Explain the difference between `ndiga`, `ndimo kwiga`, and `nize` with examples. | tense and aspect accuracy |
| T1067 | Morphology and grammar | benchmark-only | Sobanura uko `-ra-` ikoreshwa muri `Ndakora` na `Nkorera i Kigali`. | present marker, distribution |
| T1068 | Morphology and grammar | benchmark-only | Explain how negation changes `aragenda`, `yaragiye`, and `azagenda`. | negation across tenses |
| T1069 | Morphology and grammar | benchmark-only | Show how the verb agrees with the subjects in `Umwana arakina` and `Abana barakina`. | subject prefixes |
| T1070 | Morphology and grammar | benchmark-only | Explain why `ikaramu yanjye` and `ibitabo byanjye` use different possessive forms. | possessive agreement |
| T1071 | Morphology and grammar | benchmark-only | Explain the difference between `uyu`, `aba`, `iki`, and `ibi` using nouns. | demonstratives, noun classes |
| T1072 | Morphology and grammar | benchmark-only | Sobanura impamvu bavuga `mu nzu`, `ku ishuri`, na `i Kigali`. | locative choice, examples |
| T1073 | Morphology and grammar | benchmark-only | Explain the apostrophe and agreement in `ururimi rw'Ikinyarwanda`. | elision, possessive connector |
| T1074 | Morphology and grammar | benchmark-only | Compare `gukora`, `umukozi`, `akazi`, and `akorera` without claiming they have identical stems. | derivation, lexical caution |
| T1075 | Morphology and grammar | benchmark-only | Explain how an object marker works in `Ndamubona` and `Ndakibona`. | object agreement, classes |
| T1076 | Morphology and grammar | benchmark-only | Break down `tuzabafasha` and translate it naturally. | future, subject and object markers |
| T1077 | Morphology and grammar | benchmark-only | Explain the difference between a verb infinitive beginning with `ku-` and the locative word `ku`. | homography, grammatical roles |
| T1078 | Morphology and grammar | benchmark-only | Give a careful beginner explanation of noun classes using three singular-plural pairs. | correctness, accessible teaching |
| T1079 | Morphology and grammar | benchmark-only | Explain concord in `Aya mazi meza arakonje.` | demonstrative, adjective, verb agreement |
| T1080 | Morphology and grammar | benchmark-only | Explain concord in `Iki giti kinini kirakuze.` | noun-class agreement |
| T1081 | Morphology and grammar | benchmark-only | Explain how questions are formed in `Uriga?`, `Wiga iki?`, and `Wiga he?`. | question formation |
| T1082 | Morphology and grammar | benchmark-only | Compare `ndi`, `uri`, `ari`, `turi`, `muri`, and `bari` in a compact table. | copular forms, person and number |
| T1083 | Morphology and grammar | benchmark-only | Explain when `ni` is used and when a conjugated form such as `ari` is needed. | copula distinction, examples |
| T1084 | Morphology and grammar | benchmark-only | Sobanura itandukaniro rya `nagiye`, `naragiye`, na `nari nagiye`. | past tense and aspect |
| T1085 | Morphology and grammar | benchmark-only | Explain the conditional meaning in `Nubonera umwanya uzaze.` | conditional clause, natural translation |
| T1086 | Morphology and grammar | benchmark-only | Explain the relative relationship in `igitabo nasomye` and `umuntu nabonye`. | relative clauses, agreement |
| T1087 | Morphology and grammar | benchmark-only | Explain why numbers may change form after different noun classes, using two and three. | numeral concord |
| T1088 | Morphology and grammar | benchmark-only | Show how `-anjye`, `-awe`, and `-acu` agree with `inzu`, `igitabo`, and `abana`. | possessive forms, agreement |
| T1089 | Morphology and grammar | benchmark-only | Explain the difference between `kuko`, `ubwo`, and `nubwo` with short examples. | conjunction meaning, clause use |
| T1090 | Morphology and grammar | benchmark-only | A learner asks whether every word beginning with `umu-` refers to a person. Give a careful answer. | exceptions, uncertainty, morphology |
| T1091 | Orthography and pronunciation | benchmark-only | Explain the spelling difference between `cy`, `shy`, and `jy` without inventing English sound equivalents. | orthography, pronunciation caution |
| T1092 | Orthography and pronunciation | benchmark-only | How should a beginner approach the sounds written `rw` and `ny` in Kinyarwanda? | pronunciation guidance, no false precision |
| T1093 | Orthography and pronunciation | benchmark-only | Sobanura impamvu bandika `n'` muri `n'umwarimu` kandi utange urundi rugero. | elision, orthography |
| T1094 | Orthography and pronunciation | benchmark-only | Correct the spacing and capitalization if needed: `Niga ikinyarwanda mu Rwanda.` | language-name capitalization, spacing |
| T1095 | Orthography and pronunciation | benchmark-only | Explain the written difference between `Rwanda`, `U Rwanda`, and `mu Rwanda`. | orthography, locative form |
| T1096 | Orthography and pronunciation | benchmark-only | A learner asks for an exact English pronunciation of `amakuru`. Give useful but cautious guidance. | pronunciation, uncertainty |
| T1097 | Orthography and pronunciation | benchmark-only | Explain why `mb`, `nd`, and `ng` at the start of a syllable can be difficult for English speakers. | pronunciation convention, clarity |
| T1098 | Orthography and pronunciation | benchmark-only | Kosora imyandikire: `Mwaramutse neza? nitwa Eric.` | punctuation, capitalization |
| T1099 | Orthography and pronunciation | benchmark-only | Explain the apostrophe in `w'umunyeshuri`, `ry'ishuri`, and `z'abana`. | connector agreement, elision |
| T1100 | Orthography and pronunciation | benchmark-only | Give a short reading exercise that contrasts `r`, `rw`, and `ry` without using invented words. | valid vocabulary, pronunciation practice |
| T1101 | Dialogue and conversation | benchmark-only | Write a four-turn beginner conversation in which a learner asks someone to repeat slowly. | natural dialogue, classroom usefulness |
| T1102 | Dialogue and conversation | benchmark-only | Komeza iki kiganiro mu ntera enye: A: Witwa nde? B: Nitwa Mugisha. | continuity, natural follow-up |
| T1103 | Dialogue and conversation | benchmark-only | Role-play a shop conversation where the learner asks the price and says the amount is too high. | natural transaction language |
| T1104 | Dialogue and conversation | benchmark-only | Write a short conversation where one speaker asks for directions to the bus station. | direction vocabulary, turn taking |
| T1105 | Dialogue and conversation | benchmark-only | Continue naturally without changing facts: Aline studies in Huye and her brother works in Kigali. Ask her two follow-up questions. | fact consistency, questions |
| T1106 | Dialogue and conversation | benchmark-only | Have a beginner conversation about today's weather, then correct one learner mistake gently. | dialogue, correction style |
| T1107 | Dialogue and conversation | benchmark-only | Respond in Kinyarwanda to a learner who says they are nervous about speaking incorrectly. | encouragement, natural language |
| T1108 | Dialogue and conversation | benchmark-only | Create a six-turn conversation between friends planning to meet tomorrow afternoon. | future tense, time, consistency |
| T1109 | Dialogue and conversation | benchmark-only | The user says `stop stop stop` after a long answer. Respond appropriately in one short sentence. | instruction following, brevity |
| T1110 | Dialogue and conversation | benchmark-only | A learner switches from English to Kinyarwanda midway through the conversation. Continue naturally in Kinyarwanda and keep the same topic. | code switching, continuity |
| T1111 | Tutoring and exercises | benchmark-only | Teach `mu`, `ku`, and `i` with three examples and then give two unanswered practice questions. | locatives, exercise quality |
| T1112 | Tutoring and exercises | benchmark-only | Create a five-minute lesson on subject agreement for a beginner. | lesson structure, grammar accuracy |
| T1113 | Tutoring and exercises | benchmark-only | Make four fill-in-the-blank questions about singular and plural nouns, with a separate answer key. | noun classes, answer correctness |
| T1114 | Tutoring and exercises | benchmark-only | Give a translation exercise with two easy and two intermediate sentences, then provide model answers. | difficulty progression, translation accuracy |
| T1115 | Tutoring and exercises | benchmark-only | A learner repeatedly confuses `ndi` and `ni`. Give a concise explanation and one practice question. | targeted remediation, clarity |
| T1116 | Tutoring and exercises | benchmark-only | Create a short exercise that tests `kubona` versus `kureba`, including answers. | lexical distinction, grading |
| T1117 | Tutoring and exercises | benchmark-only | Teach the phrase `Ntabwo numvise` and show how to make it more polite. | negation, classroom register |
| T1118 | Tutoring and exercises | benchmark-only | Give feedback on this learner answer: `Nitwa Aline kandi ndi umunyeshuri.` | constructive feedback, correctness |
| T1119 | Tutoring and exercises | benchmark-only | Design a three-step exercise for practicing past, present, and future forms of `kugenda`. | tense accuracy, progression |
| T1120 | Tutoring and exercises | benchmark-only | Ask the learner one question at a time to practice introductions. Start with only the first question. | instruction following, brevity |
| T1121 | Culture and register | benchmark-only | Explain how respectful plural forms can be used when speaking to one older person. | register, grammatical accuracy |
| T1122 | Culture and register | benchmark-only | Is it always rude to ask `Uri nde?` Explain how context and relationship matter. | cultural nuance, uncertainty |
| T1123 | Culture and register | benchmark-only | Give a respectful way to ask an older person to come closer, without inventing a word for please. | politeness, lexical accuracy |
| T1124 | Culture and register | benchmark-only | Explain when using a first name may sound natural or overly familiar in Rwanda. Avoid universal claims. | cultural caution, nuance |
| T1125 | Culture and register | benchmark-only | Compare a casual request to a close friend with a respectful request to a teacher. | register contrast, examples |
| T1126 | Culture and register | benchmark-only | A learner wants one fixed translation for every English use of `please`. Explain why context matters. | pragmatic meaning, teaching quality |
| T1127 | Culture and register | benchmark-only | Explain whether `Muraho` is equally natural in every greeting situation. | register and context |
| T1128 | Culture and register | benchmark-only | Show how tone can change a correction from helpful to disrespectful, using safe examples. | tutoring tone, cultural care |
| T1129 | Culture and register | benchmark-only | A user asks for the one true Rwandan way to introduce yourself. Respond without overgeneralizing. | uncertainty, cultural diversity |
| T1130 | Culture and register | benchmark-only | Explain why literal translations of proverbs may lose meaning and what a careful tutor should do. | cultural caution, uncertainty |
| T1131 | Ambiguity and uncertainty | benchmark-only | `Ejo` ishobora gusobanura iki? Baza ikibazo kimwe gifasha kumenya igihe kivugwa. | time ambiguity, clarification |
| T1132 | Ambiguity and uncertainty | benchmark-only | The user asks what an unfamiliar regional expression means. You are not sure. Respond helpfully without inventing an answer. | calibrated uncertainty |
| T1133 | Ambiguity and uncertainty | benchmark-only | A learner writes only `gusoma`. Ask a useful clarification question before giving a long explanation. | ambiguity handling, brevity |
| T1134 | Ambiguity and uncertainty | benchmark-only | The user requests a proverb translation but provides only half of the proverb. What should the tutor say? | missing context, uncertainty |
| T1135 | Ambiguity and uncertainty | benchmark-only | A user asks whether a sentence is correct, but the intended tense is unclear. Ask one precise question. | clarification, tense awareness |
| T1136 | Ambiguity and uncertainty | benchmark-only | A learner gives a word that may be misspelled. Explain how you would verify it instead of guessing. | hallucination resistance |
| T1137 | Ambiguity and uncertainty | benchmark-only | The user asks for current political information in Kinyarwanda. Explain what must be checked before answering. | freshness, fact-checking behavior |
| T1138 | Ambiguity and uncertainty | benchmark-only | A learner asks for an audio pronunciation, but you can only provide text. State the limitation and still help. | capability honesty, usefulness |
| T1139 | Ambiguity and uncertainty | benchmark-only | Two fluent speakers suggest different natural translations. Explain how both might be evaluated. | variation, non-dogmatic reasoning |
| T1140 | Ambiguity and uncertainty | benchmark-only | A learner asks for a grammar rule that has exceptions. Give a cautious answer with one example and one caveat. | calibrated explanation |
| T1141 | Reading and multi-turn consistency | benchmark-only | Read and answer in Kinyarwanda: `Aline yagiye ku isoko kugura ibitoki n'amata. Agezeyo asanga amata yashize.` Ni iki ataguze, kandi kubera iki? | reading comprehension, grounded answer |
| T1142 | Reading and multi-turn consistency | benchmark-only | Read and answer in English: `Mugabo yakerewe ku ishuri kuko imodoka yari yapfuye.` Why was Mugabo late? | grounded translation, no additions |
| T1143 | Reading and multi-turn consistency | benchmark-only | Context: Keza has two sisters and one brother. Answer only this question: How many siblings does Keza have? | factual consistency, concise answer |
| T1144 | Reading and multi-turn consistency | benchmark-only | Context: Yesterday Eric studied at home. Today he is going to the library. Ask where he studied yesterday in Kinyarwanda, then answer it. | tense consistency, question formation |
| T1145 | Reading and multi-turn consistency | benchmark-only | Soma: `Imvura yatangiye saa kumi. Abana bari bamaze kugera mu rugo.` Ese abana banyagiwe mu nzira? Sobanura udahimbye. | inference restraint, uncertainty |
| T1146 | Reading and multi-turn consistency | benchmark-only | Remember these facts for your answer: My name is Diane, I live in Musanze, and I am learning Kinyarwanda. Introduce me in Kinyarwanda without changing any fact. | fact retention, translation |
| T1147 | Reading and multi-turn consistency | benchmark-only | A learner first says they are a beginner, then asks for an explanation of noun classes. Give a beginner-level answer, not an advanced lecture. | level consistency, concision |
| T1148 | Reading and multi-turn consistency | benchmark-only | The previous answer claimed that `ejo` always means tomorrow. Correct that claim explicitly and give both meanings. | self-correction, ambiguity |
| T1149 | Reading and multi-turn consistency | benchmark-only | The user has already said their name is Jonathan. Continue the conversation in Kinyarwanda without asking their name again. | conversational memory, naturalness |
| T1150 | Reading and multi-turn consistency | benchmark-only | Read: `Uwera yasabye Kalisa kumuzanira igitabo cye.` List the possible owners of the book and ask a clarifying question. | pronoun ambiguity, clarification |

## Review Rules

- All 150 prompts remain permanently held out.
- A reviewer may mark a prompt invalid before scoring its response.
- Model identities remain hidden until scoring is complete.
- A failed response can inspire a new training objective, but neither its prompt
  nor its answer may be copied into the rewrite dataset.
