# Ho va ten: Dinh Minh Quan
# MSSV: 20120355
# Lop: 20_21

class Clause():
    def fPrintClause(self):
        return ""
    def fLiteralsList(self):
        return []
    def fSymbolsList(self):
        return set()

class And(Clause):
    def __init__(self, *args):
        self.args = list(args)
    def __eq__(self, instance):
        return isinstance(instance, And) and self.args == instance.args
    def __hash__(self):
        return hash(("and", tuple(hash(arg) for arg in self.args)))
    def add(self, arg):
        self.args.append(arg)
    def fPrintClause(self):
        if len(self.args) == 1:
            return self.args[0].fPrintClause()
        return " AND ".join([arg.fPrintClause() for arg in self.args])
    def fSymbolsList(self):
        return set.union(*[arg.fSymbolsList() for arg in self.args])

class Or(Clause):
    def __init__(self, *args):
        try:
            for arg in args:
                if not isinstance(arg, Clause):
                    raise Exception("It's not a logical clause")
            self.args = list(args)
        except:
            if isinstance(args[0], list):
                self.args = args[0]
            else:
                raise Exception("There is an Error at OR's operation")

    def __eq__(self, instance):
        return isinstance(instance, Or) and self.args == instance.args
    def __hash__(self):
        return hash(("or", tuple(hash(arg) for arg in self.args)))
    def fPrintClause(self):
        if len(self.args) == 1:
            return self.args[0].fPrintClause()
        symbolsList = [arg.fPrintClause() for arg in self.args]
        symbolsList.sort()
        return " OR ".join(symbolsList)
    def fSymbolsList(self):
        return set.union(*[arg.fSymbolsList() for arg in self.args])
    def fLiteralsList(self):
        lst = []
        for arg in self.args:
            lst.append(arg)
        return lst

class Not(Clause):
    def __init__(self, o):
        self.o = o
    def __eq__(self, instance):
        return isinstance(instance, Not) and self.o == instance.o
    def __hash__(self):
        return hash(("not", hash(self.o)))
    def fPrintClause(self):
        return "-" + self.o.fPrintClause()
    def fSymbolsList(self):
        return self.o.fSymbolsList()
    def fLiteralsList(self):
        lst = [Not(self.o)]
        return lst

class Symbol(Clause):
    def __init__(self, s):
        self.s = s
    def __eq__(self, instance):
        return isinstance(instance, Symbol) and self.s == instance.s
    def __hash__(self):
        return hash(("symbol", self.s))
    def fPrintClause(self):
        return self.s
    def fSymbolsList(self):
        return {self.s}
    def fLiteralsList(self):
        lst = [Symbol(self.s)]
        return lst

def PL_RESOLUTION(baseKnowledge, alpha):
    clauses = baseKnowledge + [Not(alpha)]
    new,number,clause = [],[],[]

    while True:
        print('Available Clauses: ', end='')
        print(displayClauses(clauses))
        clausesLength = len(clauses)
        p = [(clauses[i], clauses[j]) for i in range(0,clausesLength) for j in range(i+1,clausesLength)]
        for (ci, cj) in p:
            resolvents = PL_RESOLVE(ci, cj, clauses)
            for i in resolvents:
                if not i in new:
                    new.append(i)
        check = True
        for i in new:
            if not i in clauses:
                check = False
        if check:
            return [False, number, clause]

        countClauses = 0
        for i in new:
            if not i in clauses:
                clauses.append(i)
                clause.append(displayClauses([i]))
                countClauses += 1
        number.append(countClauses)
        print(f'n Added clauses = {countClauses}')

        if '{}' in clauses:
            return [True, number, clause]

        print('\n')

def PL_RESOLVE(ci, cj, clauses):
    new = []
    ciLLst = ci.fLiteralsList()
    cjLLst = cj.fLiteralsList()

    if len(ciLLst) == 1 and len(cjLLst) == 1:
        if ciLLst[0] == Not(cjLLst[0]) or cjLLst[0] == Not(ciLLst[0]):
            new = ['{}']
    else:
        for m in ciLLst:
            for n in cjLLst:
                if m == Not(n) or Not(m) == n:
                    a = ciLLst
                    b = cjLLst
                    a.remove(m)
                    b.remove(n)
                    new_clauses = list(set(a + b))

                    check1 = False
                    if len(new_clauses) == 1:
                        new_clause = new_clauses[0]
                        check1 = True

                    elif len(new_clauses) > 1:
                        check2 = True
                        for i in new_clauses:
                            for j in new_clauses:
                                if i == Not(j) or Not(i) == j:
                                    check2 = False
                        if check2:
                            new_clause = Or(new_clauses)
                            check1 = True

                    if check1:
                        new.append(new_clause)
    
    for i in new:
        for j in clauses:
            if i == j or i == Not(Not(j)) or Not(Not(i)) == j:
                new.remove(i)
    if len(new) != 0:
        print(f'({ci.fPrintClause():<10}) and ({cj.fPrintClause():<10}) === Resolve ==>    ', end='')
        print('(' + displayClauses(new) + ')')
    return new

def symbolsSplit(txt):
    s = txt
    symbols = []
    def countOr(s):
        return s.count('OR')
    for i in range(countOr(s)):
        index = s.find('OR') - 1
        temp = s[:index]
        symbols.append(temp)
        s = s[index+4:]
    symbols.append(s)
    return symbols

def symbolsLogic(strSymb, symbString, symbLogic):
    if strSymb[0] != '-':
        index = symbString.index(strSymb)
        return symbLogic[index]
    index = symbString.index(strSymb[1:])
    return Not(symbLogic[index])
        
def displayClauses(lst):
    temp = ''
    try:
        for e in lst:
            temp += e.fPrintClause() + ' ; '
        temp = temp[:-2]
    except:
        temp = lst[0]
    return temp

#----------------------------------------------READ AND WRITE FILE---------------------------------------------

inputFiles = ['./INPUT/input_1.txt','./INPUT/input_2.txt','./INPUT/input_3.txt','./INPUT/input_4.txt','./INPUT/input_5.txt']
outputFiles = ['./OUTPUT/output_1.txt','./OUTPUT/output_2.txt','./OUTPUT/output_3.txt','./OUTPUT/output_4.txt','./OUTPUT/output_5.txt']

for i in range(len(inputFiles)):
    checkReadFile = False
    with open(inputFiles[i], 'r') as fileInput:
        lines = [l.rstrip() for l in fileInput.readlines()]
        symbString = []
        symbols = []
        for m, l in enumerate(lines):
            if m != 1:
                for symb in symbolsSplit(l):
                    if symb[0] == '-':
                        if not symb[1:] in symbols:
                            symbols.append(symb[1:])
                    else:
                        if not symb in symbols:
                            symbols.append(symb)

        symbString = symbols
        symbLogic = []
        temp = []

        for l in lines:
            temp.append(l)
        strClauses = temp
        logicalClauses = []
        
        for sym in symbString:
            symbLogic.append(Symbol(sym))

        n_kb = int(strClauses[1])
        strClauses.remove(strClauses[1])
        
        for strClause in strClauses:
            clause_logical = None
            strSymsLst = symbolsSplit(strClause)
            if len(strSymsLst) == 1:
                strSymb = strSymsLst[0]
                clause_logical = symbolsLogic(strSymb, symbString, symbLogic)
            else:
                temp_clause_list = []
                for strSymb in strSymsLst:
                    temp_clause_list.append(symbolsLogic(strSymb, symbString, symbLogic))
                clause_logical = Or(temp_clause_list)

            logicalClauses.append(clause_logical)

        alpha = logicalClauses[0]
        baseKnowledge = logicalClauses[1:]

        print('\n')
        print('************************************************************************')
        print("Base knowledge: "+ displayClauses(baseKnowledge))
        print("Alpha: "+ displayClauses([alpha]))
        print('\n')
        result = PL_RESOLUTION(baseKnowledge, alpha)
        print('\n')
        print('KB entails alpha: ',end='')
        print(result[0])
        print('************************************************************************')

        checkWriteFile = False
        with open(outputFiles[i], 'w') as fileOutput:
            j = 0
            for count in result[1]:
                fileOutput.write(f'{count}\n')
                for i in range(count):
                    fileOutput.write(f'{result[2][j]}\n')
                    j += 1
            if result[0] == True:
                fileOutput.write(f'YES')
            else:
                fileOutput.write(f'0\nNO')
            checkWriteFile = True
            fileOutput.close()

        checkReadFile = True
        fileInput.close()


