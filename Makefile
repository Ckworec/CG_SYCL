# Имя исполняемого файла
EXE = sycl-app

# Компилятор и флаги
CXX = icpx
CXXFLAGS = -O3 -fsycl -fsycl-targets=spir64_x86_64,nvptx64-nvidia-cuda

# Список исходников
SRC = $(wildcard *.cpp)

# Цель по умолчанию
all: $(EXE)

# Компиляция
$(EXE): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(EXE)

# Запуск
run: $(EXE)
	./$(EXE)

# Очистка
clean:
	rm -f $(EXE)

# Полный цикл: сборка + запуск
full: all run
