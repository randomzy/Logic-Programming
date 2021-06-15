%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% Базови предикати %%%%%%%%%%%%%%%%%%%%

append2([],L,L).
append2([A|L1],L2,[A|L]) :- append2(L1,L2,L).

member2(A,[A|_]).
member2(A,[_|T]) :- member2(A,T).

member3(A,L) :- append2(_,[A|_],L).

extract(A,[A|L],L).
extract(A,[B|L],[B|R]) :- A \= B, extract(A,L,R).

removeAll(_,[],[]).
removeAll(X,[X|L],R) :- removeAll(X,L,R).
removeAll(X,[Y|L],[Y|R]) :- X \= Y, removeAll(X,L,R).

min(A,B,A) :- A =< B.
min(A,B,B) :- A > B.

max(A,B,A) :- A >= B.
max(A,B,B) :- A < B.

min([M],M).
min([A|L],M) :- min(L,M1), min(A,M1,M).

max([M],M).
max([A|L],M) :- max(L,M1), max(A,M1,M).

between2(A,B,A) :- A =< B.
between2(A,B,C) :- A < B, A1 is A + 1, between(A1,B,C).

zip([],[],[]).
zip([A|L1],[B|L2],[[A,B]|L]) :- zip(L1,L2,L).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% Генератори %%%%%%%%%%%%%%%%%%%%%%%

%предикат който генерира всички естествени числа
nat(0).
nat(N) :- nat(N1), N is N1 + 1.

%предикат който генерира всички кратни на K
divK(_,0).
divK(K,N) :- divK(K,N1), N is K + N1.

%предикат който генерира числата на фибоначи, A e n-тото число на Фибоначи, B е n+1 число на Фибоначи
gen_fib(0,1).
gen_fib(A,B) :- gen_fib(C,A), B is C + A. 

%предикат който генерира всички двойки естествени числа
gen_pair(A,B) :- nat(N), between(0,N,A), B is N - A.

%Кодиране на естествени числа
%П(X,Y) = (2^X)*(2*Y + 1) - 1
%L(N) = X = степента на двойката в разлагането на (N+1) на прости множители
%R(N) = Y = round[(round[(N+1)/L(N)]-1)/2]

%Задача: Да се генерират всички крайни множества от естествени числа.
%Ще представим 3 решения на задачата:
%Решение 1:
%   1.Генерираме естествено число N
%   2.Генерираме списъка Ln = [1,2,...,N]
%   3.Генерираме всички подредици на Ln

%предикат, който генерира в S всички подредици на L. Подредица е редица, която може да бъде получена от друга редица като задраскаме някои от елементите й.
subsequence([],[]).
subsequence([_|L],   S ) :- subsequence(L,S).
subsequence([H|L],[H|S]) :- subsequence(L,S).

%предикат който по дадени A и B, генерира списъка L = [A, A+1, A+2, ..., B]
range(A,B,[]) :- A > B.
range(A,B,[A|L]) :- A =< B, A1 is A + 1, range(A1,B,L).

%предикат който генерира всички крайни множества от естествени числа.
gen_fin_set_1(L) :- nat(N), range(0,N,S), subsequence(S,L).

%Решение 2:
%   1.Генерираме всички двойки естествени числа m,n
%   2.1.Интерпретираме m като броя елементи, а n като сумата на елементите на списъка L = [a1,a2,...,am].
%   2.2.Генерираме всчики такива списъци, че sum i from 1 to m {ai} = n
%   3.Въвеждаме лексикографска наредба, че да няма повтарящи се елементи

%предикат който генерира всички списъци с дължина M и сума на елементите N в L.
gen_MN_list(0,0,[]).
gen_MN_list(1,M,[M]).
gen_MN_list(N,M,[H|L]) :- 
    N > 1,
    N1 is N - 1,
    between(0,M,H),
    M1 is M - H,
    gen_MN_list(N1,M1,L).

%предикат, който разпознава дали елементите на списъка L са строго растящи
strictlyIncreasing(L) :- not((append(_,[A,B|_],L), A >= B)).

%предикат който генерира всички крайни множества от естествени числа.
gen_fin_set_2(L) :-
    gen_pair(M,N),
    gen_MN_list(M,N,L),
    strictlyIncreasing(L).

%Решение 3:
%   1.Генерираме естествено число N
%   2.По дадено число N генерираме списък от позициите на които в двоичното представяне на N стои 1
%   2.Това е множеството N=b(k-1)b(k-2)...b0 -> { i | bi = 1, i < k}
%Това решение е най-доброто защото генерираме всички множества точно веднъж, и то по сравнително бърз начин.

%предикат който генерира в L позициите на които в двоичното представяне на N стои 1.
gen_ones_pos(0,_,[]).
gen_ones_pos(N,P,L) :- 
    N > 0,
    N mod 2 =:= 0,
    N1 is N div 2,
    P1 is P + 1,
    gen_ones_pos(N1,P1,L).
gen_ones_pos(N,P,[P|L]) :-
    N > 0,
    N mod 2 =:= 1,
    N1 is N div 2,
    P1 is P + 1,
    gen_ones_pos(N1,P1,L).

%предикат който генерира всички крайни множества от естествени числа.
gen_fin_set(L) :- nat(N), gen_ones_pos(N,0,L).

%%%%%%%%%%%%%%%%%%%%%%%% ВАЖНА ЗАДАЧА %%%%%%%%%%%%%%%%%%%%%%%%
%Задача: Да се генерират всички двуместни крайни релации над естествените числа
%Решение 1:
%   1.Генерираме списъка L подмножество на N
%   2.По списъка L=[a0,a1,a2,...,an]
%	генерираме всички списъци R=[[b0,c0],[b1,c1],...,[bn,cn]],
%	такива че ai=bi+ci

gen_list_pairs([],[]).
gen_list_pairs([A|T1],[[B,C]|T2]) :- 
    gen_list_pairs(T1,T2), 
    between(0,A,B), C is A - B.

gen_relations(R) :- gen_fin_set(L), gen_list_pairs(L,R).

%Решение 2:
%   1.Генерираме А подмножество на N и B подмножество на N (За да генерираме А и Б, може да използваме някое от решенията на предходната задача)
%   2.Образуваме C = AxB
%   3.Генерираме подсписъците на C

%Предикат който от списъка L=[a0,a1,...,an], и елемент X, създава нов списък R=[[X,a0],[X,a1],...,[X,an]].
makePair(_,[],[]).
makePair(X,[A|L],[[X,A]|R]) :- makePair(X,L,R).

cartesian([],_,[]).
cartesian([A|T],L,R) :- 
    cartesian(T,L,R1), 
    makePair(A,L,S), 
    append(S,R1,R).

cartesian_tuples([],[]).
cartesian_tuples([H|T], [B|R]) :- member(B, H), cartesian_tuples(T,R).

gen_relations2(R) :- 
    gen_pair(A,B),
    gen_ones_pos(A,0,AL),
    gen_ones_pos(B,0,BL),
    cartesian(AL,BL,S),
    subsequence(S,R).

%предикат който разпознава дали релацията R е транзитивна
%VaVbVc(aRb & bRc -> aRc)
%VaVbVc(not(aRb & bRc) v aRc)
%not EaEbEc(not( not(aRb & bRc) v aRc) )
%not EaEbEc(aRb & bRc & not(aRc))
notTransitive(R) :-
    member([A,B],R),
    member([B,C],R),
    not(member([A,C],R)).

%предикат който генериа всички транзитивни релации
gen_transitive_relations(R) :- 
    gen_relations(R), 
    not(notTransitive(R)).

%Задача: Да се генерират всички мултимножества над естествените числа
%Решение:
%   1.Генерираме всички двойки (m,n)
%   2.m e броя елементи, елементите на са по-големи от n

gen_powerset_MN(0,_,[]).
gen_powerset_MN(M,N,[A|T]) :-
    M > 0,
    M1 is M - 1,
    between(0,N,A),
    gen_powerset_MN(M1,N,T).

gen_powersets(L) :- gen_pair(M,N), gen_powerset_MN(M,N,L).

%Задача: Да се генерират всички думи над азбуката L
%Решение:
%   1. Генерираме дължина N
%   2. Генерираме всички думи с дължина N

gen_wordN(0,_,[]).
gen_wordN(N,L,[A|T]) :-
    N > 0,
    N1 is N - 1,
    member(A,L),
    gen_wordN(N1,L,T).

%Задача: Да се генерират всички рационални числа които попадат в окръжността с център [OX,OY] и радиус R

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% Работа с множества/списъци  %%%%%%%%%%%%%%%%%%
len([],0).
len([_|L], N) :- len(L,N1), N is N1 + 1.

% ВАЖНО!!! каква е разликата между nth_element_SINGLE_PURPOSE и nth_element:
% nth_element_SINGLE_PURPOSE не може да се използва по следния начин
% nth_element_SINGLE_PURPOSE([0,1,2,3,4,5], X, 2). т.е не може да зададем въпроса на коя
% позиция стои първото срешане на елемента 2.
nth_element_SINGLE_PURPOSE([X|_], 0, X).
nth_element_SINGLE_PURPOSE([_|L], N, X) :- N1 is N - 1, nth_element_SINGLE_PURPOS(L, N1, X).

nth_element([X|_], 0, X).
nth_element([_|L], N, X) :- nth_element(L, N1, X), N is N1 + 1.

%предикат който от списъка L създава нов списък R, такъв че има само уникални елементи от L
unique([],[]).
unique([A|L],R) :-  unique(L,R), member(A,R).
unique([A|L],[A|R]) :- unique(L,R), not(member(A,R)).

isList([]).
isList([_|_]).

%L e списък от списъци
flatten([],[]).
flatten([A|L],R) :- 
    isList(A),
    flatten(A,R1),
    flatten(L,R2),
    append(R1,R2,R).
flatten([A|L],[A|R]):-
    not(isList(A)),
    flatten(L,R).

%частен случай на flatten
join([],[]).
join([A|L], R) :- join(L,U), append(A,U,R).

%генерира в Z всички поднможества получени от разликата на X\Y
setDifference(_,_,[]).
setDifference(X,Y,[A|Z]) :-
    member(A,X), not(member(A,Y)), %A принадлежи на X\Y.
    setDifference(X,[A|Y],Z). %A вече е "забранен", затова го слагаме към Y 

setDifference2([],_,[]).
setDifference2([A|X],Y,[A|Z]) :-
    not(member(A,Y)), %A принадлежи на X\Y.
    setDifference2(X,[A|Y],Z). %A вече е "забранен", затова го слагаме към Y 
setDifference2([A|X],Y,Z) :-
    member(A,Y), %A принадлежи на X\Y.
    setDifference2(X,Y,Z). %A вече е "забранен", затова го слагаме към Y 

%предикат който намира обединението на списъка от списъци L в R
union([],[]).
union([A|L], R) :- 
    union(L,U),
    setDifference2(A,U,S),
    append(S,U,R).

in_union(X,A,B) :- member(X,A); member(X,B).
in_intersection(X,A,B) :- member(X,A), member(X,B).
in_differencr(X,A,B) :- member(X,A), not(member(X,B)).
is_subset_of(A,B) :- not(( member(X,A), not(member(X,B)) )).
set_equal(A,B) :- is_subset_of(A,B), is_subset_of(B,A).

%генерира в X всички подмножества на Y
subset(X,Y) :- setDifference(Y,[],X).

sublist_complement([],[],[]).
sublist_complement([A|L],L1,[A|L2]) :- sublist_complement(L,L1,L2).
sublist_complement([A|L],[A|L1],L2) :- sublist_complement(L,L1,L2).

%разбиване на списъка L е списък от списъци [L1,L2,...,Lk], т.ч
%   1. concat Li = L
%   2. Vi и Vj нямат елементи които стоят на едни и същи позиции в L, i != j
partition([],0,[]).
partition([A|L],1,[[A|L]]).
partition(L,K,[V|D]) :-
    K > 1,
    sublist_complement(L,V,U),
    K1 is K - 1,
    partition(U,K1,D).

split([],[],[]).
split([A],[A],[]).
split([A,B|L], [A|L1], [B|L2]) :- split(L,L1,L2).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% Графи  %%%%%%%%%%%%%%%%%%%%
