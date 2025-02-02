docker_build:
	docker build -t arm-assembly-env .     

docker_run: 
	docker run -it --rm -v "$(pwd):/workspace" arm-assembly-env 

program: program.o
	ld -o program program.o

program.o: program.s
	as -o program.o program.s

clean:
	rm -f *.o program

run: program
	./program; echo $$?