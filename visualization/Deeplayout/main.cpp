// #include "deeplayout.h"
// #include <QApplication>
// #include <QCommandLineParser>
// #include <QCommandLineOption>
// #include <QFileInfo>
// #include <QDir>

// int main(int argc, char *argv[])
// {
//     QApplication app(argc, argv);
//     QCoreApplication::setApplicationName("Deeplayout");
//     QCoreApplication::setApplicationVersion("1.0");

//     QCommandLineParser parser;
//     parser.setApplicationDescription("Floor plan visualization tool");
//     parser.addHelpOption();
//     parser.addVersionOption();

//     // Command line options
//     QCommandLineOption inputOption(QStringList() << "i" << "input",
//         "Input floor plan data file", "input");
//     QCommandLineOption outputOption(QStringList() << "o" << "output",
//         "Output visualization file", "output");
//     QCommandLineOption headlessOption(QStringList() << "headless",
//         "Run in headless mode (no GUI)");

//     parser.addOption(inputOption);
//     parser.addOption(outputOption);
//     parser.addOption(headlessOption);

//     parser.process(app);

//     if (parser.isSet(headlessOption)) {
//         if (!parser.isSet(inputOption)) {
//             fprintf(stderr, "Error: Input file must be specified in headless mode\n");
//             return 1;
//         }

//         Deeplayout processor;
//         QString inputFile = parser.value(inputOption);
//         QImage inputImage(inputFile);
        
//         if (inputImage.isNull()) {
//             fprintf(stderr, "Error: Could not load input file: %s\n", 
//                     qPrintable(inputFile));
//             return 1;
//         }

//         processor.ReadImageData(inputImage);
//         processor.HouseAbstract();

//         QString outputFile = parser.isSet(outputOption) ? 
//             parser.value(outputOption) : 
//             QDir::currentPath() + "/" + QFileInfo(inputFile).baseName() + "_output.png";
            
//         processor.SaveImageData(outputFile, false);
//         printf("Visualization saved to: %s\n", qPrintable(outputFile));
//         return 0;
//     }

//     // GUI mode
//     Deeplayout w;
//     w.showMaximized();
//     return app.exec();
// }
#include "deeplayout.h"
#include <QApplication>
#include <QCommandLineParser>
#include <QCommandLineOption>
#include <QFileInfo>
#include <QDir>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    QCoreApplication::setApplicationName("Deeplayout");
    QCoreApplication::setApplicationVersion("1.0");

    QCommandLineParser parser;
    parser.setApplicationDescription("Floor plan visualization tool");
    parser.addHelpOption();
    parser.addVersionOption();

    // Command line options
    QCommandLineOption inputOption(QStringList() << "i" << "input",
        "Input floor plan data file", "input");
    QCommandLineOption outputOption(QStringList() << "o" << "output",
        "Output visualization file", "output");
    QCommandLineOption headlessOption(QStringList() << "headless",
        "Run in headless mode (no GUI)");
    QCommandLineOption textOption(QStringList() << "t" << "text",
        "Show room labels in visualization");

    parser.addOption(inputOption);
    parser.addOption(outputOption);
    parser.addOption(headlessOption);
    parser.addOption(textOption);

    parser.process(app);

    if (parser.isSet(headlessOption)) {
        if (!parser.isSet(inputOption)) {
            fprintf(stderr, "Error: Input file must be specified in headless mode\n");
            return 1;
        }

        Deeplayout processor;
        QString inputFile = parser.value(inputOption);
        QImage inputImage(inputFile);
        
        if (inputImage.isNull()) {
            fprintf(stderr, "Error: Could not load input file: %s\n", 
                    qPrintable(inputFile));
            return 1;
        }

        processor.ReadImageData(inputImage);
        processor.HouseAbstract();

        QString outputFile = parser.isSet(outputOption) ? 
            parser.value(outputOption) : 
            QDir::currentPath() + "/" + QFileInfo(inputFile).baseName() + "_output.png";
            
        processor.SaveImageData(outputFile, parser.isSet(textOption)); // Pass textState as true
        printf("Visualization saved to: %s\n", qPrintable(outputFile));
        return 0;
    }

    // GUI mode
    Deeplayout w;
    w.showMaximized();
    return app.exec();
}