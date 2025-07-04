/*
  Minimal AudioGridder Server - Standalone Implementation
  
  This is a simplified version that can be compiled without external dependencies
  for testing and basic functionality.
*/

#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <ctime>

#ifdef _WIN32
#include <windows.h>
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#endif

class MinimalAudioGridderServer {
private:
    bool running = false;
    int port = 55055;
    int serverSocket = -1;
    
public:
    MinimalAudioGridderServer(int serverPort = 55055) : port(serverPort) {}
    
    void start() {
        std::cout << "Starting Minimal AudioGridder Server on 0.0.0.0:" << port << std::endl;
        running = true;
        
#ifdef _WIN32
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            std::cerr << "WSAStartup failed" << std::endl;
            return;
        }
        serverSocket = socket(AF_INET, SOCK_STREAM, 0);
#else
        serverSocket = socket(AF_INET, SOCK_STREAM, 0);
#endif
        
        if (serverSocket < 0) {
            std::cerr << "Failed to create socket" << std::endl;
            return;
        }
        
        // Set socket options
        int opt = 1;
#ifdef _WIN32
        setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt));
#else
        setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif
        
        // Bind to 0.0.0.0
        struct sockaddr_in address;
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY; // 0.0.0.0
        address.sin_port = htons(port);
        
        if (bind(serverSocket, (struct sockaddr*)&address, sizeof(address)) < 0) {
            std::cerr << "Bind failed on port " << port << std::endl;
            return;
        }
        
        if (listen(serverSocket, 3) < 0) {
            std::cerr << "Listen failed" << std::endl;
            return;
        }
        
        std::cout << "✓ Server successfully bound to 0.0.0.0:" << port << std::endl;
        std::cout << "✓ Server listening for connections..." << std::endl;
        std::cout << "✓ AudioGridder Minimal Server is ready!" << std::endl;
        
        // Basic server loop
        while (running) {
            std::cout << "[" << getCurrentTime() << "] Server running on 0.0.0.0:" << port << " (minimal functionality)" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(10));
        }
        
#ifdef _WIN32
        closesocket(serverSocket);
        WSACleanup();
#else
        close(serverSocket);
#endif
    }
    
    void stop() {
        running = false;
        std::cout << "Server stopped." << std::endl;
    }
    
private:
    std::string getCurrentTime() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto tm = *std::localtime(&time_t);
        
        char buffer[100];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm);
        return std::string(buffer);
    }
};

int main(int argc, char* argv[]) {
    std::cout << "AudioGridder GPU Server - Minimal Build" << std::endl;
    std::cout << "Version: 1.0.0 (Minimal)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    int port = 55055;
    if (argc > 1) {
        port = std::atoi(argv[1]);
    }
    
    MinimalAudioGridderServer server(port);
    
    try {
        server.start();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
